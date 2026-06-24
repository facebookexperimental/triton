"""Case 7 — Fused weight-gradient GEMM + bias-gradient reduction, hand-written WS.

Structurally mirrors the production best-perf reference
`geo/kernel_library/generated/bwd/addmm_1d_bias_reduce.py`
(`matmul_kernel_tma_ws_blackwell` with `FUSE_BIAS_REDUCE=True`): a Blackwell
warp-specialized persistent GEMM whose bias-gradient reduction runs in its OWN
async task, reading the same SMEM `dout` tiles the MMA consumes.

Computes the backward weight/bias gradients of a linear layer:
    dW = doutᵀ @ act    [K_out, N_in]   (GEMM, contraction dim = M)
    db = dout.sum(0)     [K_out]         (column-reduction of dout over M)

Mapping to the geo GEMM (C = A @ B):
    geo M  ← K_out   (output rows)
    geo N  ← N_in    (output cols)
    geo K  ← M       (contraction)
    A (LHS) = doutᵀ  — loaded row-major as dout [BLOCK_M, BLOCK_KO], then
              `tlx.local_trans` before the dot (column-major operand).
    B (RHS) = act    — loaded row-major as act [BLOCK_M, BLOCK_NI].
    bias    = sum of dout over its M (row) axis → [BLOCK_KO].

Four warp groups (async tasks), exactly the geo layout:
  * default      — epilogue: TMEM acc → SMEM → TMA store of dW.
  * MMA (1 warp) — `async_dot(local_trans(dout), act)` → TMEM accumulator.
  * producer     — TMA loads of dout + act into the SMEM ring.
  * bias-reduce  — `bias_acc += tl.sum(dout, axis=0)`, then stores db.

Like geo, dout's SMEM "empty" barrier has arrive_count=2: both the MMA and the
bias task consume each dout tile, so the producer must wait for both before
reusing the buffer (the MMA's arrival also frees the paired act tile).
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _bufidx_phase(cnt, NBUF: tl.constexpr):
    return cnt % NBUF, (cnt // NBUF) & 1


@triton.jit
def wgrad_bias_ws(
    DOUT,  # [M, K_out]   upstream gradient
    ACT,  # [M, N_in]    forward input activation
    DW,  # [K_out, N_in] weight-gradient output
    DB,  # [K_out]       bias-gradient output
    M,
    K_out,
    N_in,
    BLOCK_KO: tl.constexpr,
    BLOCK_NI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_ko = tl.cdiv(K_out, BLOCK_KO)
    num_ni = tl.cdiv(N_in, BLOCK_NI)
    num_tiles = num_ko * num_ni
    k_tiles = tl.cdiv(M, BLOCK_M)

    dout_desc = tl.make_tensor_descriptor(
        DOUT, [M, K_out], [K_out, 1], [BLOCK_M, BLOCK_KO]
    )
    act_desc = tl.make_tensor_descriptor(ACT, [M, N_in], [N_in, 1], [BLOCK_M, BLOCK_NI])
    dw_desc = tl.make_tensor_descriptor(
        DW, [K_out, N_in], [N_in, 1], [BLOCK_KO, BLOCK_NI]
    )

    # SMEM rings: dout tiles (A, transposed for MMA + reduced for bias) and act
    # tiles (B). TMEM accumulator ring overlaps MMA and epilogue.
    dout_smem = tlx.local_alloc((BLOCK_M, BLOCK_KO), tl.float16, NUM_SMEM_BUFFERS)
    act_smem = tlx.local_alloc((BLOCK_M, BLOCK_NI), tl.float16, NUM_SMEM_BUFFERS)
    acc_tmem = tlx.local_alloc(
        (BLOCK_KO, BLOCK_NI), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem
    )
    c_smem = tlx.local_alloc((BLOCK_KO, BLOCK_NI), tl.float16, 2)

    dout_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    act_full = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    # Shared "empty" for dout+act: consumed by BOTH the MMA (frees A and B) and
    # the bias task (frees A) → arrive_count=2 (the geo FUSE_BIAS_REDUCE idiom).
    smem_empty = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=2)
    tmem_full = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    tmem_empty = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    dsize: tl.constexpr = 2  # fp16
    dout_bytes: tl.constexpr = dsize * BLOCK_M * BLOCK_KO
    act_bytes: tl.constexpr = dsize * BLOCK_M * BLOCK_NI

    with tlx.async_tasks():
        # ── default: epilogue consumer (TMEM acc → SMEM → TMA store of dW) ──
        with tlx.async_task("default"):
            tmem_cnt = 0
            for tile_id in range(pid, num_tiles, num_programs):
                pid_ko = tile_id // num_ni
                pid_ni = tile_id % num_ni
                tbuf, trphase = _bufidx_phase(tmem_cnt, NUM_TMEM_BUFFERS)

                tlx.barrier_wait(tmem_full[tbuf], trphase)
                acc = tlx.local_load(acc_tmem[tbuf])
                tlx.barrier_arrive(tmem_empty[tbuf], 1)
                c = acc.to(tl.float16)
                tlx.local_store(c_smem[0], c)
                tlx.fence_async_shared()
                tlx.async_descriptor_store(
                    dw_desc, c_smem[0], [pid_ko * BLOCK_KO, pid_ni * BLOCK_NI]
                )
                tlx.async_descriptor_store_wait(0)
                tmem_cnt += 1

        # ── MMA consumer: async_dot(local_trans(dout), act) → TMEM ──
        with tlx.async_task(num_warps=1, num_regs=232):
            tmem_cnt = 0
            smem_cnt = 0
            for tile_id in range(pid, num_tiles, num_programs):
                tbuf, twphase = _bufidx_phase(tmem_cnt, NUM_TMEM_BUFFERS)
                # Peeled first K-iteration: acquire TMEM, use_acc=False.
                buf, phase = _bufidx_phase(smem_cnt, NUM_SMEM_BUFFERS)
                tlx.barrier_wait(dout_full[buf], phase)
                tlx.barrier_wait(act_full[buf], phase)
                tlx.barrier_wait(tmem_empty[tbuf], twphase ^ 1)
                tlx.async_dot(
                    tlx.local_trans(dout_smem[buf]),
                    act_smem[buf],
                    acc_tmem[tbuf],
                    use_acc=False,
                    mBarriers=[smem_empty[buf]],
                    out_dtype=tl.float32,
                )
                smem_cnt += 1
                for _ in range(1, k_tiles):
                    buf, phase = _bufidx_phase(smem_cnt, NUM_SMEM_BUFFERS)
                    tlx.barrier_wait(dout_full[buf], phase)
                    tlx.barrier_wait(act_full[buf], phase)
                    tlx.async_dot(
                        tlx.local_trans(dout_smem[buf]),
                        act_smem[buf],
                        acc_tmem[tbuf],
                        use_acc=True,
                        mBarriers=[smem_empty[buf]],
                        out_dtype=tl.float32,
                    )
                    smem_cnt += 1
                # Wait the last dot to drain the operands, then hand TMEM to epilogue.
                last_buf, last_phase = _bufidx_phase(smem_cnt - 1, NUM_SMEM_BUFFERS)
                tlx.barrier_wait(smem_empty[last_buf], last_phase)
                tlx.barrier_arrive(tmem_full[tbuf], 1)
                tmem_cnt += 1

        # ── producer: TMA loads of dout + act into the SMEM ring ──
        with tlx.async_task(num_warps=1, num_regs=232):
            smem_cnt = 0
            for tile_id in range(pid, num_tiles, num_programs):
                pid_ko = tile_id // num_ni
                pid_ni = tile_id % num_ni
                off_ko = pid_ko * BLOCK_KO
                off_ni = pid_ni * BLOCK_NI
                for k_idx in range(0, k_tiles):
                    off_m = k_idx * BLOCK_M
                    buf, phase = _bufidx_phase(smem_cnt, NUM_SMEM_BUFFERS)
                    tlx.barrier_wait(smem_empty[buf], phase ^ 1)
                    tlx.barrier_expect_bytes(dout_full[buf], dout_bytes)
                    tlx.async_descriptor_load(
                        dout_desc, dout_smem[buf], [off_m, off_ko], dout_full[buf]
                    )
                    tlx.barrier_expect_bytes(act_full[buf], act_bytes)
                    tlx.async_descriptor_load(
                        act_desc, act_smem[buf], [off_m, off_ni], act_full[buf]
                    )
                    smem_cnt += 1

        # ── bias-reduce consumer: db = sum(dout, axis=0), then store ──
        with tlx.async_task(num_warps=1, num_regs=232):
            smem_cnt = 0
            for tile_id in range(pid, num_tiles, num_programs):
                pid_ko = tile_id // num_ni
                pid_ni = tile_id % num_ni
                bias_acc = tl.zeros((BLOCK_KO,), dtype=tl.float32)
                for _ in range(0, k_tiles):
                    buf, phase = _bufidx_phase(smem_cnt, NUM_SMEM_BUFFERS)
                    tlx.barrier_wait(dout_full[buf], phase)
                    old = tlx.local_load(dout_smem[buf])
                    bias_acc += tl.sum(old.to(tl.float32), axis=0)
                    tlx.barrier_arrive(smem_empty[buf])
                    smem_cnt += 1
                # db is the full reduction over M for this K_out tile, identical
                # across N_in tiles — only the first N tile writes it.
                if pid_ni == 0:
                    off_ko = pid_ko * BLOCK_KO
                    tl.store(
                        DB + off_ko + tl.arange(0, BLOCK_KO),
                        bias_acc.to(DB.dtype.element_ty),
                    )
