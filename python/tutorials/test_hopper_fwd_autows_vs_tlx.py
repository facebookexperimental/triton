"""
Test: Compare Hopper autoWS FA forward against all 4 TLX reference kernels.

Runs:
  1. Accuracy comparison (autoWS vs TLX hopper_fa_ws vs PyTorch)
  2. Performance benchmark (autoWS SWP on/off vs all 4 TLX variants)

Usage:
  TRITON_USE_META_WS=1 python test_hopper_fwd_autows_vs_tlx.py
  TRITON_USE_META_WS=1 python test_hopper_fwd_autows_vs_tlx.py --bench
"""

import os
import sys
import torch
import triton
import importlib.util

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hopper():
    return torch.cuda.get_device_capability()[0] == 9


_this_dir = os.path.dirname(os.path.abspath(__file__))
_tlx_dir = os.path.join(_this_dir, "..", "..", "third_party", "tlx", "tutorials")


def _import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# TLX kernels
tlx_ws = _import("hopper_fa_ws", os.path.join(_tlx_dir, "hopper_fa_ws.py"))
tlx_pipe = _import("hopper_fa_ws_pipelined", os.path.join(_tlx_dir, "hopper_fa_ws_pipelined.py"))
tlx_pp = _import("hopper_fa_ws_pipelined_pingpong", os.path.join(_tlx_dir, "hopper_fa_ws_pipelined_pingpong.py"))
tlx_pp_persist = _import(
    "hopper_fa_ws_pipelined_pingpong_persistent",
    os.path.join(_tlx_dir, "hopper_fa_ws_pipelined_pingpong_persistent.py"),
)


def load_autows(swp=True):
    os.environ["TRITON_HOPPER_SWP"] = "1" if swp else "0"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    return _import(f"autows_swp{int(swp)}", os.path.join(_this_dir, "fused-attention-ws-device-tma-hopper.py"))


def pytorch_ref(q, k, v, sm_scale):
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


# ── Accuracy ──────────────────────────────────────────────────────────────


def test_accuracy(Z, H, N_CTX, D, dtype=torch.float16, atol=2e-2):
    torch.manual_seed(20)
    sm = 0.5
    q = torch.randn((Z, H, N_CTX, D), dtype=dtype, device=DEVICE)
    k = torch.randn((Z, H, N_CTX, D), dtype=dtype, device=DEVICE)
    v = torch.randn((Z, H, N_CTX, D), dtype=dtype, device=DEVICE)

    ref = pytorch_ref(q, k, v, sm)
    tlx_out = tlx_ws.attention(q, k, v, sm).to(dtype)
    autows = load_autows(swp=True)
    aws_out = autows.attention(q, k, v, False, sm, "ws_persistent", False, 0, False).to(dtype)

    td = (tlx_out - ref).abs().max().item()
    ad = (aws_out - ref).abs().max().item()
    at = (aws_out - tlx_out).abs().max().item()

    print(f"  Z={Z} H={H} N={N_CTX} D={D}")
    print(f"  TLX-PyTorch={td:.6f}  AutoWS-PyTorch={ad:.6f}  AutoWS-TLX={at:.6f}")

    nan = torch.isnan(aws_out).sum().item()
    if nan:
        print(f"  FAILED: {nan} NaN")
        return False
    try:
        torch.testing.assert_close(aws_out, ref, atol=atol, rtol=0)
        print("  PASSED")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


# ── Benchmark ─────────────────────────────────────────────────────────────


def bench_one(fn, warmup=5, rep=20):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def run_benchmark():
    print("\n" + "=" * 100)
    print("Performance Benchmark: AutoWS (SWP on/off) vs 4 TLX variants")
    print("=" * 100)

    aws_swp = load_autows(swp=True)
    aws_no = load_autows(swp=False)

    labels = ["AutoWS+SWP", "AutoWS-SWP", "TLX-ws", "TLX-pipe", "TLX-pp", "TLX-pp-persist"]
    header = f"{'Config':<28}" + "".join(f"{l:>14}" for l in labels)
    print(header)
    print("-" * (28 + 14 * len(labels)))

    for BATCH, H, N_CTX in [(4, 48, 2048), (4, 48, 4096), (4, 48, 8192)]:
        D = 128
        dtype = torch.float16
        q = torch.randn((BATCH, H, N_CTX, D), dtype=dtype, device=DEVICE)
        k = torch.randn((BATCH, H, N_CTX, D), dtype=dtype, device=DEVICE)
        v = torch.randn((BATCH, H, N_CTX, D), dtype=dtype, device=DEVICE)
        flops = 2 * 2.0 * BATCH * H * N_CTX * N_CTX * D
        sm = 0.5

        fns = [
            lambda: aws_swp.attention(q, k, v, False, sm, "ws_persistent", False, 0, False),
            lambda: aws_no.attention(q, k, v, False, sm, "ws_persistent", False, 0, False),
            lambda: tlx_ws.attention(q, k, v, sm),
            lambda: tlx_pipe.attention(q, k, v, sm),
            lambda: tlx_pp.attention(q, k, v, sm),
            lambda: tlx_pp_persist.attention(q, k, v, sm),
        ]

        tflops = []
        for fn in fns:
            try:
                ms = bench_one(fn)
                tflops.append(flops * 1e-12 / (ms * 1e-3))
            except Exception:
                tflops.append(0.0)

        config = f"B={BATCH} H={H} N={N_CTX} D={D}"
        vals = "".join(f"{t:>11.1f} TF" for t in tflops)
        print(f"{config:<28}{vals}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not is_hopper():
        print("Skipping: requires Hopper GPU")
        sys.exit(0)

    do_bench = "--bench" in sys.argv

    ok = True
    if not do_bench:
        print("=" * 60)
        print("Hopper AutoWS FA Forward: Accuracy Test")
        print("=" * 60)

        for Z, H, N, D in [(1, 1, 1024, 128), (1, 1, 2048, 128), (1, 1, 4096, 128), (2, 4, 1024, 128)]:
            print(f"\nTest: Z={Z} H={H} N_CTX={N} HEAD_DIM={D}")
            if not test_accuracy(Z, H, N, D):
                ok = False

        print("\n" + "=" * 60)
        print("ALL ACCURACY TESTS PASSED" if ok else "SOME ACCURACY TESTS FAILED")
        print("=" * 60)

    if do_bench:
        run_benchmark()

    sys.exit(0 if ok else 1)
