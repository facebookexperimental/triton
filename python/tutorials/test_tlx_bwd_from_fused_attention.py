"""
Test script: Compare backward kernels from fused-attention-ws-device-tma.py
(original bwd) and blackwell_fa_ws_pipelined_persistent.py (TLX bwd).

Three backward implementations are compared:
  1. PyTorch reference    — matmul-based softmax attention, autograd backward
  2. Original bwd         — _attn_bwd / _attn_bwd_persist from fused-attention-ws-device-tma.py
  3. TLX bwd              — _attn_bwd_ws from blackwell_fa_ws_pipelined_persistent.py

Both Triton backward kernels share the same forward pass so that the
comparison isolates backward-pass differences only.

The script runs:
  - Accuracy comparison: verifies dQ, dK, dV against PyTorch reference
  - Performance benchmark: measures TFLOPS for Triton autoWS vs TLX bwd
"""

import sys
import os
import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor
import importlib.util

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

    # persistent = baseVariant in ("persistent", "ws_persistent")

    def grid_persist(META):
        return (
            min(NUM_SMS,
                triton.cdiv(q.shape[2], META["BLOCK_M"]) * q.shape[0] * q.shape[1]),
            1,
            1,
        )

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True
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

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True
        if persistent:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

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

            def grid_persist_bwd(meta):
                return (
                    min(NUM_SMS,
                        triton.cdiv(N_CTX, meta["BLOCK_N1"]) * BATCH * N_HEAD),
                    1,
                    1,
                )

            _attn_bwd_persist_orig[grid_persist_bwd](
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
                maxRegAutoWS=192,
                early_tma_store_lowering=True,
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
                maxRegAutoWS=192,
                early_tma_store_lowering=True,
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

    # BWD_BLOCK_M1 = 64  # 128 or 64
    # EPILOGUE_SUBTILE = 4 if BWD_BLOCK_M1 == 128 and HEAD_DIM == 128 else 2
    # GROUP_SIZE_M = 1

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
        # BLOCK_M1=BWD_BLOCK_M1,
        # EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
        # GROUP_SIZE_M=GROUP_SIZE_M,
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
# Performance benchmark
# ============================================================================
# warmup=2000, rep=2000
def benchmark_bwd(Z, H, N_CTX, HEAD_DIM, causal, baseVariant, dtype=torch.float16, warmup=1000, rep=1000):
    """Benchmark original bwd vs TLX bwd and return (orig_ms, tlx_ms, orig_tflops, tlx_tflops)."""
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    sm_scale = 0.5

    persistent = baseVariant in ("persistent", "ws_persistent")
    tri_out, M = shared_forward(q, k, v, sm_scale, causal, baseVariant)
    dout = torch.randn_like(q)

    # Warm up both paths once to trigger compilation
    run_original_bwd(q, k, v, tri_out, M, dout, sm_scale, causal, persistent)
    run_tlx_bwd(q, k, v, tri_out, M, dout, sm_scale, causal)

    # Benchmark original bwd
    orig_ms = triton.testing.do_bench(
        lambda: run_original_bwd(q, k, v, tri_out, M, dout, sm_scale, causal, persistent),
        warmup=warmup,
        rep=rep,
    )

    # Benchmark TLX bwd
    tlx_ms = triton.testing.do_bench(
        lambda: run_tlx_bwd(q, k, v, tri_out, M, dout, sm_scale, causal),
        warmup=warmup,
        rep=rep,
    )

    # Compute TFLOPS: bwd = 2.5 * 2 * (2 * B * H * N * N * D)
    flops_per_matmul = 2.0 * Z * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul * 2.5  # 2.0(bwd) + 0.5(recompute)
    orig_tflops = total_flops * 1e-12 / (orig_ms * 1e-3)
    tlx_tflops = total_flops * 1e-12 / (tlx_ms * 1e-3)

    return orig_ms, tlx_ms, orig_tflops, tlx_tflops


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
    # TODO: TLX bwd is broken with current descriptor API, skip for now
    tlx_dq = torch.zeros_like(orig_dq)
    tlx_dk = torch.zeros_like(orig_dk)
    tlx_dv = torch.zeros_like(orig_dv)

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
    import argparse

    parser = argparse.ArgumentParser(description="Compare backward kernels for fused attention")
    parser.add_argument("--accuracy-only", action="store_true", help="Run only accuracy checks, skip benchmarks")
    args = parser.parse_args()

    if not is_blackwell():
        print("This test requires a Blackwell GPU. Skipping.")
        sys.exit(0)

    configs = [
        # (Z,  H,  N_CTX, HEAD_DIM, causal, baseVariant)
        # (8,  16, 1024,  64,  False, "ws"),
        # (8,  16, 1024,  128, False, "ws"),
        # (8, 16, 1024, 64, False, "ws_persistent"), # data race
        (8, 16, 1024, 128, False, "ws_persistent"),  # works
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

    if args.accuracy_only:
        sys.exit(0 if all_pass else 1)

    # ---- Performance benchmark -----------------------------------------------
    print(f"\n{'=' * 78}")
    print("Performance Benchmark: FA BWD — Triton autoWS vs TLX")
    print(f"{'=' * 78}\n")

    bench_configs = [
        # (Z,  H,  N_CTX, HEAD_DIM, causal, baseVariant)
        (8, 16, 1024, 128, False, "ws_persistent"),
        (8, 16, 2048, 128, False, "ws_persistent"),
        (8, 16, 4096, 128, False, "ws_persistent"),
        (4, 32, 4096, 128, False, "ws_persistent"),
    ]

    cw = [8, 6, 8, 10, 16, 14, 14, 14, 10]
    header = ["Z", "H", "N_CTX", "HEAD_DIM", "baseVariant", "Triton (ms)", "TLX (ms)", "Triton TFLOPS", "Speedup"]
    sep = ["-" * (w - 1) for w in cw]
    print_table([header, sep], cw)

    for Z, H, N_CTX, HEAD_DIM, causal, baseVariant in bench_configs:
        orig_ms, tlx_ms, orig_tflops, tlx_tflops = benchmark_bwd(
            Z,
            H,
            N_CTX,
            HEAD_DIM,
            causal,
            baseVariant,
        )
        speedup = tlx_ms / orig_ms if orig_ms > 0 else float("inf")
        row = [
            str(Z),
            str(H),
            str(N_CTX),
            str(HEAD_DIM),
            baseVariant,
            f"{orig_ms:.3f}",
            f"{tlx_ms:.3f}",
            f"{orig_tflops:.1f}",
            f"{speedup:.2f}x",
        ]
        print_table([row], cw)

    print(f"\n{'=' * 78}")
    print("Benchmark complete.")
    print(f"{'=' * 78}")
