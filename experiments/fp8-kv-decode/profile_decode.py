"""Single-workload driver for rocprofv3; setup and compilation precede repeats."""

import argparse
import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]


def load_kernel():
    path = ROOT / "third_party/tlx/tutorials/amd_pa_decode_fp8.py"
    spec = importlib.util.spec_from_file_location("tlx_fp8_profile", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=2304)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--page", type=int, default=64)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--fp8-q", action="store_true")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--backend", choices=["tlx", "aiter"], default="tlx")
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--shared-inputs", action="store_true")
    args = parser.parse_args()
    if args.backend == "aiter" or args.shared_inputs:
        from types import SimpleNamespace
        from bench_decode import inputs, make_backend, reference
        a = SimpleNamespace(**vars(args), tokens=1, scales="shared", causal=True, block_m=16)
        x = inputs(a)
        expected = reference(x, a)
        fn = make_backend("aiter_unified" if args.backend == "aiter" else "tlx_" + args.variant, x, a)
        result = fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), expected, atol=5e-4, rtol=1e-2)
        for _ in range(2):
            result = fn()
            torch.cuda.synchronize()
        print("profile workload", vars(args), "finite", bool(torch.isfinite(result).all()))
        return
    module = load_kernel()
    q, k, v, lengths, table, ks, vs, qs, max_len, _, _ = module._make_test_inputs([args.length], hq=args.heads,
                                                                                  hkv=args.kv_heads, dim=args.dim,
                                                                                  page=args.page, fp8_q=args.fp8_q)
    for _ in range(3):
        kwargs = {"variant": args.variant, "block_n": args.block_n, "num_warps": args.warps}
        result = module.fp8_kv_decode(q, k, v, lengths, table, ks, vs, max_context_len=max_len, q_scale=qs,
                                      num_splits=args.splits, window_size=args.window, **kwargs)
        torch.cuda.synchronize()
    print("profile workload", vars(args), "finite", bool(torch.isfinite(result).all()))


if __name__ == "__main__":
    main()
