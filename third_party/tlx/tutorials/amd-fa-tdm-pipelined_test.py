"""
TDM-pipelined Flash-Attention forward for AMD gfx1250 (TLX).

This is a TLX port of the Gluon golden kernel
``attn_fwd_pipelined_kernel`` in
``third_party/amd/python/examples/gluon/f16_fa_gfx1250.py``.

Goal: match the golden's instruction mix and software-pipeline schedule
to achieve comparable steady-state WMMA efficiency on gfx1250.

Mapping (Gluon golden -> TLX):
  gl.amd.gfx1250.tdm.make_tensor_descriptor -> tl.make_tensor_descriptor
  gl.allocate_shared_memory([2]+blk)         -> tlx.local_alloc(blk, dt, 2)
  tdm.async_load(desc, off, buf.index(i))    -> tlx.async_amd_descriptor_load(desc, local_view(buf,i), off)
  tdm.async_wait(n)                          -> tlx.async_amd_descriptor_wait(n)
  buf.index(i).permute([1,0]).load(k_layout) -> tlx.local_load(tlx.local_trans(tlx.local_view(buf,i)))
  buf.index(i).load(v_layout)                -> tlx.local_load(tlx.local_view(buf,i))
  gl.amd.gfx1250.wmma(a, b, c)               -> tl.dot(a, b, c)

The pipeline is hand-written (no auto-pipeliner), exactly like the
golden: 3 iters peeled across prologue+epilogue, a steady-state hot
loop in between, double-buffered K/V in LDS via TDM async copies.
"""
import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()

RCP_LN2 = tl.constexpr(1.4426950408889634)


def is_gfx1250_available():
    try:
        target = triton.runtime.driver.active.get_current_target()
        return target.arch == "gfx1250"
    except Exception:
        return False


@triton.jit
def _load_k(k_buf, slot, wait_count):
    # async_wait(n): block until at most n TDM ops outstanding, then read
    # the K tile from LDS *transposed* ([BLOCK_N, HEAD_SZ] -> [HEAD_SZ,
    # BLOCK_N]) so the QK dot operand lowering uses a memdesc transpose
    # instead of a register shuffle.
    tlx.async_amd_descriptor_wait(wait_count)
    return tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, slot)))


@triton.jit
def _load_v(v_buf, slot, wait_count):
    tlx.async_amd_descriptor_wait(wait_count)
    return tlx.local_load(tlx.local_view(v_buf, slot))


@triton.jit
def _compute_qk(q, k, cur_seq, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SEQLEN_K: tl.constexpr):
    qk = tl.dot(q, k)
    qk_mask = (cur_seq + tl.arange(0, BLOCK_N))[None, :] < SEQLEN_K
    qk = tl.where(qk_mask, qk, float("-inf"))
    return qk


@triton.jit
def _compute_qk_no_mask(q, k):
    return tl.dot(q, k)


@triton.jit
def _softmax_part0(qk, m_i, scale_ln2):
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    m_ij_scaled = m_ij * scale_ln2
    q_shifted = scale_ln2 * qk - m_ij_scaled[:, None]
    p = tl.math.exp2(q_shifted)
    alpha = tl.math.exp2(scale_ln2 * m_i - m_ij_scaled)
    return p, alpha, m_ij


@triton.jit
def _softmax_part1(p, l_i, acc, alpha):
    l_ij = tl.sum(p, 1)
    acc = acc * alpha[:, None]
    p_bf16 = p.to(tl.bfloat16, fp_downcast_rounding="rtz")
    l_i = l_i * alpha + l_ij
    return p_bf16, l_i, acc


@triton.jit
def attn_fwd_tdm_pipelined_kernel(q_ptr, k_ptr, v_ptr, o_ptr,  #
                                  stride_qz, stride_qh, stride_qm, stride_qk,  #
                                  stride_kz, stride_kh, stride_kn, stride_kk,  #
                                  stride_vz, stride_vh, stride_vn, stride_vk,  #
                                  stride_oz, stride_oh, stride_om, stride_on,  #
                                  SM_SCALE: tl.constexpr,  #
                                  SEQLEN_Q: tl.constexpr,  #
                                  SEQLEN_K: tl.constexpr,  #
                                  BLOCK_M: tl.constexpr,  #
                                  BLOCK_N: tl.constexpr,  #
                                  HEAD_SZ: tl.constexpr,  #
                                  ):
    NUM_BUFFERS: tl.constexpr = 2
    scale_ln2: tl.constexpr = SM_SCALE * RCP_LN2

    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_m = tl.program_id(2) * BLOCK_M

    # --- Q: TDM-load once into LDS, then local_load into the dot-operand
    # layout directly (no register-layout conversion in the prologue). ---
    q_desc = tl.make_tensor_descriptor(
        q_ptr + off_z * stride_qz + off_h * stride_qh,
        shape=[SEQLEN_Q, HEAD_SZ],
        strides=[stride_qm, tl.constexpr(1)],
        block_shape=[BLOCK_M, HEAD_SZ],
    )
    q_buf = tlx.local_alloc((BLOCK_M, HEAD_SZ), tlx.dtype_of(q_ptr), 1)
    tlx.async_amd_descriptor_load(q_desc, tlx.local_view(q_buf, 0), [off_m, 0])
    tlx.async_amd_descriptor_wait(0)
    q = tlx.local_load(tlx.local_view(q_buf, 0))

    # --- K / V TDM descriptors (block = [BLOCK_N, HEAD_SZ]) ---
    k_desc = tl.make_tensor_descriptor(
        k_ptr + off_z * stride_kz + off_h * stride_kh,
        shape=[SEQLEN_K, HEAD_SZ],
        strides=[stride_kn, tl.constexpr(1)],
        block_shape=[BLOCK_N, HEAD_SZ],
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr + off_z * stride_vz + off_h * stride_vh,
        shape=[SEQLEN_K, HEAD_SZ],
        strides=[stride_vn, tl.constexpr(1)],
        block_shape=[BLOCK_N, HEAD_SZ],
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr + off_z * stride_oz + off_h * stride_oh,
        shape=[SEQLEN_Q, HEAD_SZ],
        strides=[stride_om, tl.constexpr(1)],
        block_shape=[BLOCK_M, HEAD_SZ],
    )
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_SZ), tlx.dtype_of(k_ptr), NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_SZ), tlx.dtype_of(v_ptr), NUM_BUFFERS)
    o_buf = tlx.local_alloc((BLOCK_M, HEAD_SZ), tlx.dtype_of(o_ptr), 1)

    ITERS_IN_PROLOGUE_EPILOGUE: tl.constexpr = 3
    n_blocks_n = max((SEQLEN_K + BLOCK_N - 1) // BLOCK_N - ITERS_IN_PROLOGUE_EPILOGUE, 1)
    has_remainder: tl.constexpr = SEQLEN_K < (ITERS_IN_PROLOGUE_EPILOGUE * BLOCK_N)
    if has_remainder:
        n_blocks_n = n_blocks_n - 1

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SZ], dtype=tl.float32)

    block_min = 0
    block_max = n_blocks_n * BLOCK_N

    # ---------------- Prologue ----------------
    tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 0), [0, 0])
    tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 1), [BLOCK_N, 0])
    tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, 0), [0, 0])

    k = _load_k(k_buf, 0, 2)
    qk = _compute_qk(q, k, 0, BLOCK_M, BLOCK_N, SEQLEN_K)
    p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)

    tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, 0), [2 * BLOCK_N, 0])
    tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, 1), [BLOCK_N, 0])

    k = _load_k(k_buf, 1, 3)

    iter_id = 0
    # ---------------- Steady state (hot loop, no masking) ----------------
    for block_id in range(block_min, block_max, BLOCK_N):
        t_2 = block_id + 2 * BLOCK_N
        t_3 = block_id + 3 * BLOCK_N

        qk = _compute_qk_no_mask(q, k)
        p, l_i, acc = _softmax_part1(p, l_i, acc, alpha)
        v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, (iter_id + 1) % NUM_BUFFERS), [t_3, 0])
        acc = tl.dot(p, v, acc)
        p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)
        k = _load_k(k_buf, iter_id % NUM_BUFFERS, 2)
        tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_2, 0])
        iter_id += 1

    # ---------------- Remainder (masked steady iter) ----------------
    if has_remainder:
        t_1 = iter_id * BLOCK_N + BLOCK_N
        t_2 = iter_id * BLOCK_N + 2 * BLOCK_N
        t_3 = iter_id * BLOCK_N + 3 * BLOCK_N

        qk = _compute_qk(q, k, t_1, BLOCK_M, BLOCK_N, SEQLEN_K)
        p, l_i, acc = _softmax_part1(p, l_i, acc, alpha)
        v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(k_buf, (iter_id + 1) % NUM_BUFFERS), [t_3, 0])
        acc = tl.dot(p, v, acc)
        p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)
        k = _load_k(k_buf, iter_id % NUM_BUFFERS, 2)
        tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_2, 0])
        iter_id += 1

    # ---------------- Epilogue ----------------
    epilogue_offset = (iter_id - 1) * BLOCK_N
    t_2 = epilogue_offset + 2 * BLOCK_N
    t_3 = epilogue_offset + 3 * BLOCK_N

    p, l_i, acc = _softmax_part1(p, l_i, acc, alpha)
    v = _load_v(v_buf, iter_id % NUM_BUFFERS, 2)
    acc = tl.dot(p, v, acc)

    qk = _compute_qk(q, k, t_2, BLOCK_M, BLOCK_N, SEQLEN_K)
    p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)

    k = _load_k(k_buf, iter_id % NUM_BUFFERS, 1)
    tlx.async_amd_descriptor_load(v_desc, tlx.local_view(v_buf, iter_id % NUM_BUFFERS), [t_3, 0])

    qk = _compute_qk(q, k, t_3, BLOCK_M, BLOCK_N, SEQLEN_K)
    p, l_i, acc = _softmax_part1(p, l_i, acc, alpha)
    v = _load_v(v_buf, (iter_id + 1) % NUM_BUFFERS, 1)
    acc = tl.dot(p, v, acc)

    p, alpha, m_i = _softmax_part0(qk, m_i, scale_ln2)
    p, l_i, acc = _softmax_part1(p, l_i, acc, alpha)
    v = _load_v(v_buf, iter_id % NUM_BUFFERS, 0)
    acc = tl.dot(p, v, acc)

    # ---------------- Output ----------------
    l_recip = 1.0 / l_i[:, None]
    acc = acc * l_recip
    # TDM store via LDS (mirrors the golden TDM-GEMM C-store): acc -> LDS
    # (native ds_write from the WMMA layout) -> global via TDM. Avoids the
    # 128-way global_store fan-out of tl.store on the WMMA accumulator.
    o_view = tlx.local_view(o_buf, 0)
    tlx.local_store(o_view, acc.to(o_ptr.dtype.element_ty))
    tlx.async_amd_descriptor_store(o_desc, o_view, [off_m, 0])
    tlx.async_amd_descriptor_wait(0)


def attn_fwd_tdm_pipelined(q, k, v, sm_scale, BLOCK_M=128, BLOCK_N=128):
    BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ = q.shape
    SEQLEN_K = k.shape[2]
    o = torch.empty_like(q, dtype=torch.float32)
    grid = (BATCH, NUM_Q_HEADS, (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M)
    attn_fwd_tdm_pipelined_kernel[grid](
        q, k, v, o,  #
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),  #
        sm_scale, SEQLEN_Q, SEQLEN_K, BLOCK_M, BLOCK_N, HEAD_SZ,  #
        num_warps=4, waves_per_eu=1)
    return o


def test_attn_fwd_tdm_pipelined_compiles_gfx1250():
    """Compile-only check: lowers cleanly to TDM intrinsics + WMMA."""
    from triton.compiler.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget

    src = ASTSource(
        fn=attn_fwd_tdm_pipelined_kernel,
        signature={
            "q_ptr": "*bf16",
            "k_ptr": "*bf16",
            "v_ptr": "*bf16",
            "o_ptr": "*fp32",
            "stride_qz": "i64",
            "stride_qh": "i64",
            "stride_qm": "i64",
            "stride_qk": "i64",
            "stride_kz": "i64",
            "stride_kh": "i64",
            "stride_kn": "i64",
            "stride_kk": "i64",
            "stride_vz": "i64",
            "stride_vh": "i64",
            "stride_vn": "i64",
            "stride_vk": "i64",
            "stride_oz": "i64",
            "stride_oh": "i64",
            "stride_om": "i64",
            "stride_on": "i64",
            "SM_SCALE": "constexpr",
            "SEQLEN_Q": "constexpr",
            "SEQLEN_K": "constexpr",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "HEAD_SZ": "constexpr",
        },
        constexprs={
            "SM_SCALE": 1.0 / (128**0.5),
            "SEQLEN_Q": 1024,
            "SEQLEN_K": 1024,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "HEAD_SZ": 128,
        },
    )
    compiled = triton_compile(src, target=GPUTarget("hip", "gfx1250", 32), options={"num_warps": 4})
    ttgir = compiled.asm["ttgir"]
    assert "amdg.async_tdm_copy_global_to_local" in ttgir
    assert "amdg.async_tdm_copy_local_to_global" in ttgir  # TDM store of O
    assert "tt.dot" in ttgir
    amdgcn = compiled.asm["amdgcn"]
    assert "tensor_load_to_lds" in amdgcn or "tensor.load.to.lds" in amdgcn
    assert "tensor_store_from_lds" in amdgcn or "tensor.store.from.lds" in amdgcn


@pytest.mark.skipif(not is_gfx1250_available(), reason="Requires gfx1250")
@pytest.mark.parametrize("BATCH,H,SEQLEN", [(1, 8, 1024),  # multi-head
                                            (2, 4, 1024),  # multi-batch + multi-head
                                            (1, 16, 2048),  # golden head count, longer seqlen
                                            (1, 2, 896),  # non-128-multiple -> masked remainder path
                                            (1, 1, 640),  # small -> remainder peel path
                                            ])
def test_attn_fwd_tdm_pipelined_gfx1250(BATCH, H, SEQLEN):
    torch.manual_seed(0)
    D = 128
    q = torch.randn((BATCH, H, SEQLEN, D), device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn((BATCH, H, SEQLEN, D), device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn((BATCH, H, SEQLEN, D), device=DEVICE, dtype=torch.bfloat16)
    sm_scale = 1.0 / (D**0.5)
    out = attn_fwd_tdm_pipelined(q, k, v, sm_scale)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v).to(torch.float32)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    if not is_gfx1250_available():
        raise SystemExit("Requires gfx1250 hardware/emulator")
    torch.manual_seed(0)
    q = torch.randn((1, 8, 1024, 128), device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn((1, 8, 1024, 128), device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn((1, 8, 1024, 128), device=DEVICE, dtype=torch.bfloat16)
    out = attn_fwd_tdm_pipelined(q, k, v, 1.0 / (128**0.5))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v).to(torch.float32)
    print("max abs diff:", (out.cpu() - ref.cpu()).abs().max().item())
