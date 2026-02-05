import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hopper
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]


configs = [
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_KV": 2,
            "NUM_MMA_WARPS": 8,
            "NUM_MMA_GROUPS": 2,
        },
        num_stages=0,
        num_warps=4,
        pre_hook=_host_descriptor_pre_hook,
    ),
]


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS):
    bufIdx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return bufIdx, phase


@triton.jit
def _compute_offsets(tile_idx, H, num_pid_n, num_pid_in_group, N_CTX, BLOCK_M: tl.constexpr):
    group_id = tile_idx // num_pid_in_group
    first_pid_n = group_id
    start_m = tile_idx % num_pid_in_group
    off_hz = first_pid_n
    off_z = off_hz // H
    off_h = off_hz % H
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    lo, hi = 0, N_CTX
    kv_offset_y = offset_y + lo
    return start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y


@triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"])
@triton.jit
def _attn_fwd_ws_pipelined_pingpong_persistent(sm_scale, M,  #
                                               Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                                               HEAD_DIM: tl.constexpr,  #
                                               BLOCK_M: tl.constexpr,  #
                                               BLOCK_N: tl.constexpr,  #
                                               FP8_OUTPUT: tl.constexpr,  #
                                               NUM_BUFFERS_Q: tl.constexpr,  #
                                               NUM_BUFFERS_KV: tl.constexpr,  #
                                               NUM_MMA_WARPS: tl.constexpr,  #
                                               NUM_MMA_GROUPS: tl.constexpr,  #
                                               ):
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    # Compute bytes per element for each tensor type
    Q_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    K_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
    V_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v))

    # Persistent kernel setup
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = Z * H
    num_pid_in_group = num_pid_m
    total_tiles = num_pid_m * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    # allocate buffers
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    k_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    v_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS_KV)

    # allocate barriers
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q, arrive_count=1)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q, arrive_count=1)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=NUM_MMA_GROUPS)
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=1)
    v_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=NUM_MMA_GROUPS)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=1)

    with tlx.async_tasks():
        # producer group (default) - loads Q, K, V
        with tlx.async_task("default"):
            accum_cnt_kv = 0

            for i in range(0, tiles_per_sm):
                # compute offsets for this tile
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_idx, H, num_pid_n, num_pid_in_group, N_CTX, BLOCK_M)

                # load q0
                q_bufIdx, q_phase = _get_bufidx_phase(i, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_bufIdx], Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM)
                qo_offset_y_split = qo_offset_y
                tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx], [qo_offset_y_split, 0], q_fulls[q_bufIdx])

                kv_offset = kv_offset_y + lo
                kv_bufIdx, kv_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)

                # load K
                tlx.barrier_wait(k_empties[kv_bufIdx], kv_phase ^ 1)
                tlx.barrier_expect_bytes(k_fulls[kv_bufIdx], K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                tlx.async_descriptor_load(desc_k, k_tiles[kv_bufIdx], [kv_offset, 0], k_fulls[kv_bufIdx])

                # load q1
                q_bufIdx_1 = q_bufIdx + NUM_BUFFERS_Q
                tlx.barrier_wait(q_empties[q_bufIdx_1], q_phase ^ 1)
                tlx.barrier_expect_bytes(q_fulls[q_bufIdx_1], Q_BYTES_PER_ELEM * BLOCK_M_SPLIT * HEAD_DIM)
                qo_offset_y_split = qo_offset_y + BLOCK_M_SPLIT
                tlx.async_descriptor_load(desc_q, q_tiles[q_bufIdx_1], [qo_offset_y_split, 0], q_fulls[q_bufIdx_1])

                # load V
                tlx.barrier_wait(v_empties[kv_bufIdx], kv_phase ^ 1)
                tlx.barrier_expect_bytes(v_fulls[kv_bufIdx], V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                tlx.async_descriptor_load(desc_v, v_tiles[kv_bufIdx], [kv_offset, 0], v_fulls[kv_bufIdx])
                accum_cnt_kv += 1

                # loop over K, V tiles
                for kv_idx in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                    kv_offset = kv_offset_y + kv_idx
                    kv_bufIdx, kv_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)

                    # load K
                    tlx.barrier_wait(k_empties[kv_bufIdx], kv_phase ^ 1)
                    tlx.barrier_expect_bytes(k_fulls[kv_bufIdx], K_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                    tlx.async_descriptor_load(desc_k, k_tiles[kv_bufIdx], [kv_offset, 0], k_fulls[kv_bufIdx])

                    # load V
                    tlx.barrier_wait(v_empties[kv_bufIdx], kv_phase ^ 1)
                    tlx.barrier_expect_bytes(v_fulls[kv_bufIdx], V_BYTES_PER_ELEM * BLOCK_N * HEAD_DIM)
                    tlx.async_descriptor_load(desc_v, v_tiles[kv_bufIdx], [kv_offset, 0], v_fulls[kv_bufIdx])

                    accum_cnt_kv += 1

                tile_idx += num_progs

        # Consumer group - replicated for pingpong pattern
        #
        # PINGPONG SYNCHRONIZATION OVERVIEW:
        # ----------------------------------
        # Two consumer replicas (cid=0 and cid=1) share the same WGMMA (Warp Group MMA)
        # hardware resources. To avoid resource contention, they must issue async_dot
        # operations in a coordinated "pingpong" fashion - one after the other, never
        # simultaneously.
        #
        # Named barriers 9 and 10 are used to orchestrate this:
        #   - Barrier 9: Consumer 1 signals → Consumer 0 waits
        #   - Barrier 10: Consumer 0 signals → Consumer 1 waits
        #
        # The pattern ensures:
        #   1. Consumer 0 issues its async_dot first
        #   2. Consumer 1 waits until Consumer 0 is done, then issues its async_dot
        #   3. This alternating pattern continues throughout the K-loop
        #
        # The 256 in barrier arrive/wait represents the number of threads participating
        # (8 warps * 32 threads = 256).
        #
        with tlx.async_task(num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS, registers=232, replicate=NUM_MMA_GROUPS):
            accum_cnt_kv = 0
            cid: tl.constexpr = tlx.async_task_replica_id()

            # Initial synchronization: Consumer 1 signals first to let Consumer 0 start
            # This bootstraps the pingpong pattern by ensuring Consumer 0 can proceed
            if cid == 1:
                tlx.named_barrier_arrive(9, 256)

            for i in range(0, tiles_per_sm):
                # compute offsets for this tile
                start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(
                    tile_idx, H, num_pid_n, num_pid_in_group, N_CTX, BLOCK_M)

                # initialize pointer to m and l
                m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
                acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)

                # load scales
                qk_scale = sm_scale
                qk_scale *= 1.44269504  # 1/log(2)

                # wait for the Q buffer to be populated by the producer
                q_bufIdx, q_phase = _get_bufidx_phase(i, NUM_BUFFERS_Q)
                tlx.barrier_wait(q_fulls[q_bufIdx + cid * NUM_BUFFERS_Q], q_phase)

                k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)

                # wait for the K[0] buffer to be populated by the producer
                tlx.barrier_wait(k_fulls[k_bufIdx], k_phase)

                # -- compute qk[0] ----
                k_tile = tlx.local_trans(k_tiles[k_bufIdx])

                # PINGPONG SYNC: Ensure only one consumer issues async_dot at a time
                # Consumer 0 goes first, then Consumer 1
                if cid == 0:
                    # Consumer 0 waits for Consumer 1 to be ready (prevents both issuing simultaneously)
                    tlx.named_barrier_wait(9, 256)
                else:
                    # Consumer 1 waits for Consumer 0 to finish its async_dot
                    tlx.named_barrier_wait(10, 256)

                qk = tlx.async_dot(q_tiles[q_bufIdx + cid * NUM_BUFFERS_Q], k_tile)

                if cid == 0:
                    # Consumer 0 done, signal Consumer 1 to proceed
                    tlx.named_barrier_arrive(10, 256)
                else:
                    # Consumer 1 done, signal Consumer 0 for next iteration
                    tlx.named_barrier_arrive(9, 256)

                qk = tlx.async_dot_wait(0, qk)
                # release the K buffer
                tlx.barrier_arrive(k_empties[k_bufIdx], 1)

                # -- compute m_i and l_i ----
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
                p = tl.math.exp2(qk)
                # -- compute correction factor
                alpha = tl.math.exp2(m_i - m_ij)
                # -- update output accumulator[0] --
                acc = acc * alpha[:, None]
                l_ij = tl.sum(p, 1)
                l_i = l_i * alpha + l_ij
                m_i = m_ij
                accum_cnt_kv += 1

                # loop over k, v and update accumulator
                for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)

                    # wait for the K buffer to be populated by the producer
                    tlx.barrier_wait(k_fulls[k_bufIdx], k_phase)

                    # compute qk for the current iteration
                    k_tile = tlx.local_trans(k_tiles[k_bufIdx])

                    # PINGPONG SYNC: Same pattern as first QK dot
                    # Consumer 0 goes first, Consumer 1 waits, then they swap roles
                    if cid == 0:
                        # Consumer 0 waits for Consumer 1 to be ready (prevents both issuing simultaneously)
                        tlx.named_barrier_wait(9, 256)
                    else:
                        # Consumer 1 waits for Consumer 0 to finish its async_dot
                        tlx.named_barrier_wait(10, 256)

                    qk = tlx.async_dot(q_tiles[q_bufIdx + cid * NUM_BUFFERS_Q], k_tile)

                    if cid == 0:
                        # Consumer 0 done, signal Consumer 1 to proceed
                        tlx.named_barrier_arrive(10, 256)
                    else:
                        # Consumer 1 done, signal Consumer 0 for next iteration
                        tlx.named_barrier_arrive(9, 256)

                    # compute pv from the previous iteration
                    # wait for the previous V buffer to be populated by the producer
                    v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv - 1, NUM_BUFFERS_KV)
                    tlx.barrier_wait(v_fulls[v_bufIdx], v_phase)
                    # prepare p and v for the dot
                    p = p.to(tlx.dtype_of(desc_k))
                    acc = tlx.async_dot(p, v_tiles[v_bufIdx], acc)

                    # wait for the current qk MMA to complete
                    qk = tlx.async_dot_wait(1, qk)
                    # release the K buffer
                    tlx.barrier_arrive(k_empties[k_bufIdx], 1)

                    # -- compute m_i and l_i ----
                    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                    qk = qk * qk_scale - m_ij[:, None]
                    p = tl.math.exp2(qk)
                    # -- compute correction factor
                    alpha = tl.math.exp2(m_i - m_ij)
                    l_ij = tl.sum(p, 1)
                    # update m_i and l_i
                    l_i = l_i * alpha + l_ij
                    m_i = m_ij

                    # -- update output accumulator --
                    # wait for the previous pv MMA to complete
                    acc = tlx.async_dot_wait(0, acc)
                    # release the V buffer
                    tlx.barrier_arrive(v_empties[v_bufIdx], 1)
                    acc = acc * alpha[:, None]
                    accum_cnt_kv += 1

                # compute pv from the last iteration
                # wait for the V buffer to be populated by the producer
                v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv - 1, NUM_BUFFERS_KV)
                tlx.barrier_wait(v_fulls[v_bufIdx], v_phase)
                # prepare p and v for the dot
                p = p.to(tlx.dtype_of(desc_k))
                acc = tlx.async_dot(p, v_tiles[v_bufIdx], acc)

                # signal Q empty
                acc = tlx.async_dot_wait(1, acc)
                tlx.barrier_arrive(q_empties[q_bufIdx + cid * NUM_BUFFERS_Q], 1)

                # wait for the MMA using to complete
                acc = tlx.async_dot_wait(0, acc)
                # release the V buffer
                tlx.barrier_arrive(v_empties[v_bufIdx], 1)

                # epilogue
                qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                m_i += tl.math.log2(l_i)
                acc = acc / l_i[:, None]
                offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                m_ptrs = M + off_hz * N_CTX + offs_m
                tl.store(m_ptrs, m_i)
                desc_o.store([qo_offset_y_split, 0], acc.to(tlx.dtype_of(desc_o)))

                tile_idx += num_progs


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        if q.dtype == torch.float8_e5m2:
            desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1], block_shape=dummy_block)
        else:
            desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        def grid(META):
            return (min(NUM_SMS, triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1]), 1, 1)

        ctx.grid = grid
        _attn_fwd_ws_pipelined_pingpong_persistent[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o


attention = _attention.apply


@pytest.mark.parametrize("Z", [8])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("N_CTX", [1024])
@pytest.mark.parametrize("HEAD_DIM", [128])
@pytest.mark.parametrize("mode", ["fwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"])
@pytest.mark.skipif(
    not is_hopper(),
    reason="Requires Hopper GPU",
)
def test_op(Z, H, N_CTX, HEAD_DIM, mode, provider, dtype=torch.float16):
    if mode == "bwd":
        pytest.skip("Backward pass not supported.")
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    ref_out = torch.matmul(p, v).half()
    # triton implementation
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, sm_scale).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = False
BATCH, N_HEADS, HEAD_DIM = 4, 32, 128
# vary seq length for fixed head and batch=4
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="provider",
        line_vals=["triton-fp16"] + (["flash"] if HAS_FLASH else []),
        line_names=["Triton [FP16]"] + (["Flash-2"] if HAS_FLASH else []),
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name=f"fused-attention-ws-pipelined-pingpong-persistent-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "HEAD_DIM": HEAD_DIM,
            "mode": "fwd",
        },
    ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    if is_hopper():
        print("Running benchmarks...")
        bench_flash_attention.run(print_data=True)
    else:
        print("Skipping benchmarks, no Hopper GPU found.")
