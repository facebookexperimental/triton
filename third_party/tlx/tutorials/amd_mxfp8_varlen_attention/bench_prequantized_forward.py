"""Compare prepared-input MXFP8 attention before and after optimization.

Q/K/V preparation and work-schedule construction are excluded from timing.
Both runners receive identical prepared operands and use the same exponent
mode. Effective K/V TB/s is a logical byte metric, not measured bandwidth.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from types import ModuleType

import torch
import triton

import aiter_baseline_repro as harness


def load_forward(directory, name):
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    hashes = {}
    for part in (
        "mxfp8_attention_core_gfx950",
        "mxfp8_varlen_pingpong_gfx950",
    ):
        path = directory / f"{part}.py"
        spec = importlib.util.spec_from_file_location(
            f"{name}.{part}", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(package, part, module)
        hashes[part] = hashlib.sha256(path.read_bytes()).hexdigest()
    return module, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in harness.PRODUCTION_SHAPES],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=15)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    directory = Path(__file__).resolve().parent
    baseline_directory = args.baseline_root
    if not (
        baseline_directory / "mxfp8_varlen_pingpong_gfx950.py"
    ).exists():
        baseline_directory /= (
            "third_party/tlx/tutorials/amd_mxfp8_varlen_attention"
        )
    baseline, baseline_hashes = load_forward(
        baseline_directory, "_mxfp8_bench_baseline"
    )
    candidate, candidate_hashes = load_forward(
        directory, "_mxfp8_bench_candidate"
    )
    if not hasattr(baseline, "forward_prequantized"):
        raise RuntimeError(
            "The baseline must provide the initial prepared-Q entry point"
        )
    flusher = harness.L2Flusher(256)
    rows = []
    report = {
        "completed": False,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": str(torch.cuda.get_device_properties(0)),
        "baseline_hashes": baseline_hashes,
        "candidate_hashes": candidate_hashes,
        "protocol": vars(args)
        | {
            "baseline_root": str(args.baseline_root),
            "json": str(args.json),
        },
        "byte_metric": (
            "logical main-pass K/V payload and scale bytes; excludes Q, "
            "output, metadata, redundant prefetch and cache effects"
        ),
        "results": rows,
    }
    for case in harness.PRODUCTION_SHAPES:
        if args.case and case.name not in args.case:
            continue
        torch.manual_seed(args.seed)
        q_lengths = [
            16 * value
            for value in harness.lengths(
                432, case.total_q // 16, case.max_q // 16, 3
            )
        ]
        k_lengths = harness.lengths(
            432, case.total_kv, case.max_kv, 5
        )
        cu_q, cu_k = (
            harness.cumulative(q_lengths),
            harness.cumulative(k_lengths),
        )
        q = torch.randn(
            (case.total_q, 4, 128),
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn(
            (case.total_kv, 4, 128),
            device="cuda",
            dtype=torch.bfloat16,
        )
        v = torch.randn_like(k)
        meta = candidate._metadata(
            cu_q, cu_k, case.total_q, case.total_kv, 128
        )
        state = candidate.prepare_kv(k, v, cu_k, meta)
        q8, q_scales = candidate.prepare_q(q, cu_q)
        schedule = candidate.prepare_work_schedule(
            q_lengths, k_lengths, q.device
        )
        fast_exp = case.family == "long"

        def before():
            return baseline.forward_prequantized(
                q8,
                q_scales,
                *state,
                cu_q,
                cu_k,
                harness.SOFTMAX_SCALE,
                total_kv=case.total_kv,
                block_m=128,
                num_ctas=512,
                fast_exp=fast_exp,
            )

        def after():
            return candidate.forward_prequantized(
                q8,
                q_scales,
                *state,
                cu_q,
                cu_k,
                harness.SOFTMAX_SCALE,
                total_kv=case.total_kv,
                fast_exp=fast_exp,
                work_schedule=schedule,
            )

        before.name, after.name = "prepared_baseline", "optimized"
        expected, expected_lse = before()
        actual, actual_lse = after()
        bitwise_equal = torch.equal(
            expected, actual
        ) and torch.equal(expected_lse, actual_lse)
        if not bitwise_equal:
            raise RuntimeError(
                f"Unexpected numerical change in {case.name}"
            )
        times = harness.benchmark_interleaved(
            [before, after],
            flusher,
            args.warmups,
            args.samples,
            args.rounds,
        )
        key_visits = sum(
            triton.cdiv(nq, 128)
            * triton.cdiv(nk, 256)
            * 256
            for nq, nk in zip(q_lengths, k_lengths)
            if nk
        )
        kv_bytes = 4 * (2 * 128 + 8) * key_visits
        throughput = {
            name: kv_bytes / data["timing"]["median_ms"] / 1e9
            for name, data in times.items()
        }
        rows.append(
            {
                "case": case.name,
                "family": case.family,
                "timings": times,
                "kv_bytes": kv_bytes,
                "effective_kv_TBps": throughput,
                "bitwise_equal": bitwise_equal,
                "speedup": (
                    times["prepared_baseline"]["timing"]["median_ms"]
                    / times["optimized"]["timing"]["median_ms"]
                ),
            }
        )
        print(
            f"{case.name}: {throughput['prepared_baseline']:.3f} -> "
            f"{throughput['optimized']:.3f} effective K/V TB/s",
            flush=True,
        )
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        del expected, expected_lse, actual, actual_lse
        torch.cuda.empty_cache()
    report["completed"] = True
    report["families"] = {
        family: {
            "baseline_effective_kv_TBps": statistics.geometric_mean(
                row["effective_kv_TBps"]["prepared_baseline"]
                for row in rows
                if row["family"] == family
            ),
            "optimized_effective_kv_TBps": statistics.geometric_mean(
                row["effective_kv_TBps"]["optimized"]
                for row in rows
                if row["family"] == family
            ),
            "speedup": statistics.geometric_mean(
                row["speedup"]
                for row in rows
                if row["family"] == family
            ),
        }
        for family in sorted({row["family"] for row in rows})
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["families"], indent=2))


if __name__ == "__main__":
    main()
