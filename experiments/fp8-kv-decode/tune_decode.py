"""Numerically gated TLX configuration sweep; one GPU process at a time."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from bench_decode import inputs, make_backend, reference, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=2304)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--fp8-q", action="store_true")
    parser.add_argument("--scales", default="per-head")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = SimpleNamespace(**vars(args), page=64, causal=args.tokens == 1, splits=1, block_n=64, block_m=16, warps=4)
    x = inputs(a)
    oracle = reference(x, a)
    settings = []
    if args.tokens == 1:
        for bn in (32, 64, 128):
            for splits in (1, 4, 12, 36, 72):
                if splits > (min(args.length, args.window or args.length) + bn - 1) // bn:
                    continue
                for warps in (2, 4):
                    settings.append(("tlx_vector", bn, 16, splits, warps))
    for bn in (32, 64, 128):
        split_values = (1, 4, 12, 24, 36, 72) if args.length <= 8192 else (24, 48, 96, 128)
        for splits in split_values:
            if splits > (min(args.length, args.window or args.length) + bn - 1) // bn:
                continue
            for warps in (2, 4):
                for bm in ((16, ) if args.tokens == 1 else (16, 32, 64)):
                    settings.append(("tlx_matrix", bn, bm, splits, warps))
                    if args.fp8_q and args.dim == 64:
                        settings.append(("tlx_matrix_fp8", bn, bm, splits, warps))
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for backend, bn, bm, splits, warps in settings:
        a.block_n, a.block_m, a.splits, a.warps = bn, bm, splits, warps
        row = dict(backend=backend, block_n=bn, block_m=bm, splits=splits, warps=warps)
        try:
            fn = make_backend(backend, x, a)
            actual = fn()
            err = actual.float() - oracle
            row.update(max_abs_error=float(err.abs().max()), rms_error=float(err.square().mean().sqrt()))
            torch.testing.assert_close(actual.float(), oracle, atol=5e-4, rtol=1e-2)
            row.update(status="pass", **timing(fn, 5, 10))
        except Exception as exc:
            row.update(status="failed", reason=str(exc))
        records.append(row)
        args.output.write_text(
            json.dumps({"workload": vars(args) | {"output": str(args.output)}, "results": records}, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    passing = sorted((r for r in records if r["status"] == "pass"), key=lambda r: r["median_us"])
    print("BEST", json.dumps(passing[:5]), flush=True)


if __name__ == "__main__":
    main()
