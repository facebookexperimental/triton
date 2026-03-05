"""
Test script: Compare backward kernels from fused-attention-ws-device-tma.py
(original bwd) and blackwell_fa_ws_pipelined_persistent.py (TLX bwd).

Three backward implementations are compared:
  1. PyTorch reference    — matmul-based softmax attention, autograd backward
  2. Original bwd         — _attn_bwd / _attn_bwd_persist from fused-attention-ws-device-tma.py
  3. TLX bwd              — _attn_bwd_ws from blackwell_fa_ws_pipelined_persistent.py

Both Triton backward kernels share the same forward pass so that the
comparison isolates backward-pass differences only.
"""

import sys
import os
import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


# ---------------------------------------------------------------------------
# Module imports (hyphens in filename → importlib spec_from_file_location)
# ---------------------------------------------------------------------------
import importlib.util

_this_dir = os.path.dirname(os.path.abspath(__file__))


def _import_from_file(module_name, filepath):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fused_attn_mod = _import_from_file(
    "fused_attention_ws_device_tma",
    os.path.join(_this_dir, "fused-attention-ws-device-tma.py"),
)

tlx_tutorial_path = os.path.join(
    _this_dir,
    "..",
    "..",
    "third_party",
    "tlx",
    "tutorials",
)
tlx_mod = _import_from_file(
    "blackwell_fa_ws_pipelined_persistent",
    os.path.join(tlx_tutorial_path, "blackwell_fa_ws_pipelined_persistent.py"),
)

# --- Original bwd kernels & helpers ----------------------------------------
_attn_bwd_orig = fused_attn_mod._attn_bwd
_attn_bwd_persist_orig = fused_attn_mod._attn_bwd_persist
_attn_bwd_preprocess_orig = fused_attn_mod._attn_bwd_preprocess
torch_dtype_to_triton = fused_attn_mod.torch_dtype_to_triton

# --- TLX bwd kernel & helpers ---------------------------------------------
_attn_bwd_ws_tlx = tlx_mod._attn_bwd_ws
_attn_bwd_preprocess_tlx = tlx_mod._attn_bwd_preprocess


# ============================================================================
# Shared forward — identical for both bwd variants so that the forward output,
# M (log-sum-exp), and saved tensors are exactly the same.
# ============================================================================
def shared_forward(q, k, v, sm_scale, causal, baseVariant):
    """Run the fused-attention fwd kernel and return (o, M)."""
    HEAD_DIM_K = q.shape[-1]
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]),
        device=q.device,
        dtype=torch.float32,
    )

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    warp_specialize = True
    extra_kern_args = {}
    if is_blackwell() and warp_specialize:
        extra_kern_args["maxnreg"] = 128

    persistent = baseVariant in ("persistent", "ws_persistent")

    def grid_persist(META):
        return (
            min(NUM_SMS,
                triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1]),
            1,
            1,
        )

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    if True:  # persistent: fwd non-persistent is not working yet.
        fused_attn_mod._attn_fwd_persist[grid_persist](
            sm_scale,
            M,
            q.shape[0],
            q.shape[1],
            q,
            k,
            v,
            o,
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,
            STAGE=stage,
            warp_specialize=warp_specialize,
            OUTER_LOOP=True,
            dtype=torch_dtype_to_triton(q.dtype),
            SUBTILING=False,
            VECT_MUL=0,
            FADD2_REDUCE=False,
            **extra_kern_args,
        )
    else:
        fused_attn_mod._attn_fwd[grid](
            sm_scale,
            M,
            q.shape[0],
            q.shape[1],
            q,
            k,
            v,
            o,
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,
            STAGE=stage,
            warp_specialize=warp_specialize,
            dtype=torch_dtype_to_triton(q.dtype),
            SUBTILING=False,
            VECT_MUL=0,
            FADD2_REDUCE=False,
            **extra_kern_args,
        )
    return o, M


# ============================================================================
# Original backward  (from fused-attention-ws-device-tma.py)
# ============================================================================
def run_original_bwd(q, k, v, o, M, do, sm_scale, causal, persistent):
    """Run _attn_bwd / _attn_bwd_persist and return (dq, dk, dv)."""
    assert do.is_contiguous()
    dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    BATCH, N_HEAD, N_CTX = q.shape[:3]
    HEAD_DIM = q.shape[-1]
    PRE_BLOCK = 128
    BLK_SLICE_FACTOR = 2
    RCP_LN2 = 1.4426950408889634
    arg_k = k * (sm_scale * RCP_LN2)
    assert N_CTX % PRE_BLOCK == 0

    pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
    delta = torch.empty_like(M)
    _attn_bwd_preprocess_orig[pre_grid](
        o,
        do,
        delta,
        BATCH,
        N_HEAD,
        N_CTX,
        BLOCK_M=PRE_BLOCK,
        HEAD_DIM=HEAD_DIM,
    )

    warp_specialize = True
    dummy_block = [1, 1]

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    if persistent:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        def grid_persist_bwd(meta):
            return (
                min(NUM_SMS,
                    triton.cdiv(N_CTX, meta["BLOCK_N1"]) * BATCH * N_HEAD),
                1,
                1,
            )

        _attn_bwd_persist_orig[grid_persist_bwd](
            q,
            arg_k,
            v,
            sm_scale,
            do,
            dq,
            dk,
            dv,
            M,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            BATCH,
            N_HEAD,
            N_CTX,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=HEAD_DIM,
            dtype=torch_dtype_to_triton(q.dtype),
            warp_specialize=warp_specialize,
        )
    else:
        if supports_host_descriptor():
            desc_q = TensorDescriptor(q, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                      block_shape=dummy_block)
            desc_k = TensorDescriptor(arg_k, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                      block_shape=dummy_block)
            desc_v = TensorDescriptor(v, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                      block_shape=dummy_block)
            desc_do = TensorDescriptor(do, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=dummy_block)
            desc_dq = TensorDescriptor(dq, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=dummy_block)
            desc_dk = TensorDescriptor(dk, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=dummy_block)
            desc_dv = TensorDescriptor(dv, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=dummy_block)
        else:
            desc_q, desc_k, desc_v = q, arg_k, v
            desc_do, desc_dq, desc_dk, desc_dv = do, dq, dk, dv

        def grid(meta):
            return (
                triton.cdiv(N_CTX, meta["BLOCK_N1"]),
                1,
                BATCH * N_HEAD,
            )

        _attn_bwd_orig[grid](
            desc_q,
            desc_k,
            desc_v,
            sm_scale,
            desc_do,
            desc_dq,
            desc_dk,
            desc_dv,
            M,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            BATCH,
            N_HEAD,
            N_CTX,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=HEAD_DIM,
            dtype=torch_dtype_to_triton(q.dtype),
            warp_specialize=warp_specialize,
        )

    return dq, dk, dv


# ============================================================================
# TLX backward  (from blackwell_fa_ws_pipelined_persistent.py)
# ============================================================================
def run_tlx_bwd(q, k, v, o, M, do, sm_scale, causal):
    """Run _attn_bwd_ws (TLX) and return (dq, dk, dv)."""
    assert do.is_contiguous()
    dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    BATCH, N_HEAD, N_CTX = q.shape[:3]
    HEAD_DIM = q.shape[-1]
    PRE_BLOCK = 128
    BLK_SLICE_FACTOR = 2
    RCP_LN2 = 1.4426950408889634
    arg_k = k * (sm_scale * RCP_LN2)
    assert N_CTX % PRE_BLOCK == 0

    pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
    delta = torch.empty_like(M)
    # TLX _attn_bwd_preprocess takes (O, DO, Delta, N_CTX, …)
    _attn_bwd_preprocess_tlx[pre_grid](
        o,
        do,
        delta,
        N_CTX,
        BLOCK_M=PRE_BLOCK,
        HEAD_DIM=HEAD_DIM,
    )

    dummy_block = [1, 1]
    desc_q = TensorDescriptor(q, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                              block_shape=dummy_block)
    desc_k = TensorDescriptor(arg_k, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                              block_shape=dummy_block)
    desc_v = TensorDescriptor(v, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                              block_shape=dummy_block)
    desc_do = TensorDescriptor(do, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                               block_shape=dummy_block)
    desc_dq = TensorDescriptor(dq, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                               block_shape=dummy_block)
    desc_dk = TensorDescriptor(dk, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                               block_shape=dummy_block)
    desc_dv = TensorDescriptor(dv, shape=[BATCH * N_HEAD * N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                               block_shape=dummy_block)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    stage = 3 if causal else 1
    BWD_BLOCK_M1 = 64
    EPILOGUE_SUBTILE = 4 if BWD_BLOCK_M1 == 128 and HEAD_DIM == 128 else 2
    GROUP_SIZE_M = 1

    def grid_persistent(meta):
        return (
            min(NUM_SMS,
                triton.cdiv(N_CTX, meta["BLOCK_N1"]) * BATCH * N_HEAD),
            1,
            1,
        )

    # TLX _attn_bwd_ws signature: … H, Z, N_CTX  (Z = BATCH)
    _attn_bwd_ws_tlx[grid_persistent](
        desc_q,
        desc_k,
        desc_v,
        sm_scale,
        desc_do,
        desc_dq,
        desc_dk,
        desc_dv,
        M,
        delta,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        N_HEAD,
        BATCH,
        N_CTX,
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
        HEAD_DIM=HEAD_DIM,
        STAGE=stage,
        BLOCK_M1=BWD_BLOCK_M1,
        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return dq, dk, dv


# ============================================================================
# PyTorch reference
# ============================================================================
def pytorch_reference_fwd_bwd(q, k, v, sm_scale, causal, dtype, dout):
    """Return (ref_out, ref_dq, ref_dk, ref_dv)."""
    N_CTX = q.shape[2]
    mask = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, mask == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(p, v).half()
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    return ref_out, ref_dq, ref_dk, ref_dv


# ============================================================================
# Pretty-print helpers
# ============================================================================
def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def _check(name, got, ref, atol=1e-2):
    err = _max_abs(got, ref)
    ok = err <= atol
    tag = "PASS" if ok else "FAIL"
    return tag, err


def print_table(rows, col_widths):
    """Print a fixed-width table."""
    for row in rows:
        line = ""
        for val, w in zip(row, col_widths):
            line += str(val).ljust(w)
        print(line)


# ============================================================================
# Debug: element-wise dk analysis
# ============================================================================
def _debug_dk(orig_dk, tlx_dk, ref_dk, sm_scale):
    """Dump element-wise dk comparison to find systematic scaling error."""
    RCP_LN2 = 1.4426950408889634
    LN2 = 0.6931471805599453

    o = orig_dk.float()
    t = tlx_dk.float()
    r = ref_dk.float()

    print(f"\n{'=' * 78}")
    print("  DEBUG: element-wise dK analysis")
    print(f"{'=' * 78}")

    # --- basic stats ----------------------------------------------------------
    print(f"\n  Shape: {list(r.shape)}")
    print(f"  ref_dk   — min={r.min().item():.6f}  max={r.max().item():.6f}  absmax={r.abs().max().item():.6f}")
    print(f"  orig_dk  — min={o.min().item():.6f}  max={o.max().item():.6f}  absmax={o.abs().max().item():.6f}")
    print(f"  tlx_dk   — min={t.min().item():.6f}  max={t.max().item():.6f}  absmax={t.abs().max().item():.6f}")

    # --- error location -------------------------------------------------------
    diff = (o - r).abs()
    flat_idx = diff.argmax().item()
    idx = []
    rem = flat_idx
    for s in reversed(r.shape):
        idx.append(rem % s)
        rem //= s
    idx = tuple(reversed(idx))
    print(f"\n  Max |orig-ref| error location: {idx}")
    print(f"    orig_dk = {o[idx].item():.8f}")
    print(f"    tlx_dk  = {t[idx].item():.8f}")
    print(f"    ref_dk  = {r[idx].item():.8f}")
    print(f"    error   = {diff[idx].item():.8f}")

    # --- element-wise ratio ---------------------------------------------------
    # Mask out near-zero elements to avoid division noise
    mask = r.abs() > 1e-4
    num_valid = mask.sum().item()
    print(f"\n  Ratio analysis (|ref_dk| > 1e-4,  {num_valid} / {r.numel()} elements):")

    if num_valid > 0:
        ratio_orig = o[mask] / r[mask]
        ratio_tlx = t[mask] / r[mask]

        print(f"    orig/ref — mean={ratio_orig.mean().item():.8f}  "
              f"median={ratio_orig.median().item():.8f}  "
              f"std={ratio_orig.std().item():.8f}")
        print(f"    tlx/ref  — mean={ratio_tlx.mean().item():.8f}  "
              f"median={ratio_tlx.median().item():.8f}  "
              f"std={ratio_tlx.std().item():.8f}")

        # ratio of orig to tlx (both are kernel outputs)
        mask2 = t.abs() > 1e-4
        num_valid2 = (mask & mask2).sum().item()
        if num_valid2 > 0:
            ratio_ot = o[mask & mask2] / t[mask & mask2]
            print(f"    orig/tlx — mean={ratio_ot.mean().item():.8f}  "
                  f"median={ratio_ot.median().item():.8f}  "
                  f"std={ratio_ot.std().item():.8f}")

    # --- check common scaling factors ----------------------------------------
    print("\n  Known constants:")
    print(f"    sm_scale        = {sm_scale}")
    print(f"    RCP_LN2         = {RCP_LN2:.10f}")
    print(f"    LN2             = {LN2:.10f}")
    print(f"    sm_scale*RCP_LN2= {sm_scale * RCP_LN2:.10f}")

    print("\n  Hypothesis tests (orig ≈ ref * C?):")
    for name, C in [
        ("LN2", LN2),
        ("RCP_LN2", RCP_LN2),
        ("sm_scale", sm_scale),
        ("1/sm_scale", 1.0 / sm_scale),
        ("sm_scale*RCP_LN2", sm_scale * RCP_LN2),
        ("sm_scale*LN2", sm_scale * LN2),
        ("1/(sm_scale*RCP_LN2)", 1.0 / (sm_scale * RCP_LN2)),
        ("sm_scale^2", sm_scale**2),
        ("sm_scale^2*RCP_LN2", sm_scale**2 * RCP_LN2),
    ]:
        err = (o - r * C).abs().max().item()
        print(f"    orig ≈ ref * {name:22s}  →  max|err| = {err:.6e}")

    # --- per-tile analysis (N-tile × batch-head) -----------------------------
    Z, Nh, N_CTX, D = r.shape
    BLOCK_N = 128  # BLOCK_N1
    n_tiles = N_CTX // BLOCK_N
    print(f"\n  Per N-tile max|orig-ref| and max|orig| (BLOCK_N={BLOCK_N}, {n_tiles} tiles):")
    print(f"  {'tile':>4s}  {'max|orig-ref|':>14s}  {'max|orig|':>12s}  {'max|ref|':>12s}  {'frac_zero':>10s}")
    for t_idx in range(n_tiles):
        sl = slice(t_idx * BLOCK_N, (t_idx + 1) * BLOCK_N)
        o_tile = o[:, :, sl, :]
        r_tile = r[:, :, sl, :]
        tile_err = (o_tile - r_tile).abs().max().item()
        tile_omax = o_tile.abs().max().item()
        tile_rmax = r_tile.abs().max().item()
        frac_zero = (o_tile.abs() < 1e-6).float().mean().item()
        print(f"  {t_idx:4d}  {tile_err:14.6e}  {tile_omax:12.6e}  {tile_rmax:12.6e}  {frac_zero:10.4f}")

    # --- per batch-head zero fraction -----------------------------------------
    per_bh_zero = (o.abs() < 1e-6).float().reshape(Z * Nh, -1).mean(dim=1)
    n_all_zero = (per_bh_zero > 0.99).sum().item()
    n_partial = ((per_bh_zero > 0.01) & (per_bh_zero < 0.99)).sum().item()
    n_all_ok = (per_bh_zero < 0.01).sum().item()
    print(f"\n  Per batch-head analysis ({Z * Nh} total heads):")
    print(f"    All-zero heads (>99%% zeros): {n_all_zero}")
    print(f"    Partial heads:                {n_partial}")
    print(f"    Fully written heads (<1%% z): {n_all_ok}")

    # --- accuracy of non-zero elements ----------------------------------------
    nonzero_mask = o.abs() > 1e-6
    num_nonzero = nonzero_mask.sum().item()
    print(f"\n  Non-zero element accuracy ({num_nonzero} / {o.numel()} elements):")
    if num_nonzero > 0:
        nonzero_err = (o[nonzero_mask] - r[nonzero_mask]).abs()
        print(f"    max|err| among non-zero orig = {nonzero_err.max().item():.6e}")
        print(f"    mean|err| among non-zero orig = {nonzero_err.mean().item():.6e}")
        # ratio of non-zero orig to ref
        valid = nonzero_mask & (r.abs() > 1e-4)
        if valid.sum() > 0:
            r_nz = o[valid] / r[valid]
            print(f"    ratio orig/ref — mean={r_nz.mean().item():.8f}  "
                  f"median={r_nz.median().item():.8f}  "
                  f"std={r_nz.std().item():.8f}")

    # --- per batch-head per N-tile pattern ------------------------------------
    # Show first 4 heads: which N-tiles are zero vs non-zero
    print("\n  Per-head N-tile zero pattern (Z=zeros, .=ok):")
    for bh in range(min(32, Z * Nh)):
        b, h = bh // Nh, bh % Nh
        pattern = ""
        for t_idx in range(n_tiles):
            sl = slice(t_idx * BLOCK_N, (t_idx + 1) * BLOCK_N)
            fz = (o[b, h, sl, :].abs() < 1e-6).float().mean().item()
            if fz > 0.95:
                pattern += "Z"
            elif fz < 0.05:
                pattern += "."
            else:
                pattern += "p"
        print(f"    head [{b:2d},{h:2d}]: {pattern}")

    print(f"{'=' * 78}\n")


# ============================================================================
# Main comparison
# ============================================================================
def compare_accuracy(Z, H, N_CTX, HEAD_DIM, causal, baseVariant, dtype=torch.float16, atol=1e-2):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    sm_scale = 0.5
    dout = torch.randn_like(q)

    # ---- 1. PyTorch reference ------------------------------------------------
    ref_out, ref_dq, ref_dk, ref_dv = pytorch_reference_fwd_bwd(
        q,
        k,
        v,
        sm_scale,
        causal,
        dtype,
        dout,
    )

    # ---- 2. Shared Triton forward --------------------------------------------
    persistent = baseVariant in ("ws_persistent")
    tri_out, M = shared_forward(q, k, v, sm_scale, causal, baseVariant)
    tri_out_half = tri_out.half()

    # ---- 3. Original bwd from fused-attention-ws-device-tma.py ---------------
    orig_dq, orig_dk, orig_dv = run_original_bwd(
        q,
        k,
        v,
        tri_out,
        M,
        dout,
        sm_scale,
        causal,
        persistent,
    )

    # ---- 4. TLX bwd from blackwell_fa_ws_pipelined_persistent.py -------------
    tlx_dq, tlx_dk, tlx_dv = run_tlx_bwd(
        q,
        k,
        v,
        tri_out,
        M,
        dout,
        sm_scale,
        causal,
    )

    # ---- Debug: element-wise dk analysis -------------------------------------
    # _debug_dk(orig_dk, tlx_dk, ref_dk, sm_scale)

    # ---- Print header --------------------------------------------------------
    hdr = f"Config: Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}, causal={causal}, baseVariant={baseVariant}"
    print(f"\n{'=' * 78}")
    print(hdr)
    print(f"{'=' * 78}")

    # ---- Forward accuracy (should be identical; same kernel) ------------------
    fwd_tag, fwd_err = _check("fwd", tri_out_half, ref_out, atol)
    print(f"\n  Forward vs Reference        max|err|={fwd_err:.6e}  [{fwd_tag}]")

    # ---- Backward accuracy table ---------------------------------------------
    #
    #  Columns:  Gradient | orig vs ref | tlx vs ref | orig vs tlx
    #
    cw = [12, 28, 28, 28]  # column widths
    header = ["Gradient", "Original vs Reference", "TLX vs Reference", "Original vs TLX"]
    sep = ["-" * (w - 2) for w in cw]

    print(f"\n  Backward accuracy (max |err|, atol={atol}):\n")
    print_table([header, sep], cw)

    results = {}
    for name, orig_g, tlx_g, ref_g in [
        ("dQ", orig_dq, tlx_dq, ref_dq),
        ("dK", orig_dk, tlx_dk, ref_dk),
        ("dV", orig_dv, tlx_dv, ref_dv),
    ]:
        orig_tag, orig_err = _check(name, orig_g, ref_g, atol)
        tlx_tag, tlx_err = _check(name, tlx_g, ref_g, atol)
        cross_tag, cross_err = _check(name, orig_g, tlx_g, atol)

        row = [
            name,
            f"{orig_err:.6e}  [{orig_tag}]",
            f"{tlx_err:.6e}  [{tlx_tag}]",
            f"{cross_err:.6e}  [{cross_tag}]",
        ]
        print_table([row], cw)

        results[f"{name}_orig"] = orig_tag
        results[f"{name}_tlx"] = tlx_tag
        results[f"{name}_cross"] = cross_tag

    results["fwd"] = fwd_tag

    # ---- Summary line --------------------------------------------------------
    all_ok = all(v == "PASS" for v in results.values())
    print(f"\n  Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return results


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    if not is_blackwell():
        print("This test requires a Blackwell GPU. Skipping.")
        sys.exit(0)

    configs = [
        # (Z,  H,  N_CTX, HEAD_DIM, causal, baseVariant)
        #(8,  16, 1024,  64,  False, "ws"),
        #(8,  16, 1024,  128, False, "ws"),
        #(8, 16, 1024, 64, False, "ws_persistent"), # data race
        (8, 16, 1024, 128, False, "ws_persistent"), # works
    ]

    all_pass = True
    for Z, H, N_CTX, HEAD_DIM, causal, baseVariant in configs:
        results = compare_accuracy(Z, H, N_CTX, HEAD_DIM, causal, baseVariant)
        if any(v != "PASS" for v in results.values()):
            all_pass = False

    print(f"\n{'=' * 78}")
    if all_pass:
        print("*** ALL CONFIGURATIONS PASSED ***")
    else:
        print("*** SOME CONFIGURATIONS FAILED ***")
    print(f"{'=' * 78}")
