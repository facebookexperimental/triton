"""Compare prepared forward with #4683's BF16-Q, prepared-KV core.

Preparation is excluded from timing. Effective K/V TB/s counts logical
main-pass K/V payload and scale bytes; it is not measured HBM bandwidth.
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
    for part in ("mxfp8_attention_core_gfx950", "mxfp8_varlen_pingpong_gfx950"):
        path = directory / f"{part}.py"
        spec = importlib.util.spec_from_file_location(f"{name}.{part}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(package, part, module)
        hashes[part] = hashlib.sha256(path.read_bytes()).hexdigest()
    return module, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=[c.name for c in harness.PRODUCTION_SHAPES])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=15)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    directory = Path(__file__).resolve().parent
    baseline_directory = args.baseline_root
    if not (baseline_directory / "mxfp8_varlen_pingpong_gfx950.py").exists():
        baseline_directory /= "third_party/tlx/tutorials/amd_mxfp8_varlen_attention"
    baseline, baseline_hashes = load_forward(baseline_directory, "_mxfp8_bench_baseline")
    candidate, candidate_hashes = load_forward(directory, "_mxfp8_bench_candidate")
    flusher = harness.L2Flusher(256)
    rows = []
    report = {
        "completed": False,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": str(torch.cuda.get_device_properties(0)),
        "baseline_hashes": baseline_hashes,
        "candidate_hashes": candidate_hashes,
        "protocol": vars(args) | {"baseline_root": str(args.baseline_root), "json": str(args.json)},
        "byte_metric":
        "logical main-pass K/V payload and scale bytes; excludes Q, output, metadata, redundant prefetch and cache effects",
        "results": rows,
    }
    for case in harness.PRODUCTION_SHAPES:
        if args.case and case.name not in args.case:
            continue
        torch.manual_seed(args.seed)
        q_lengths = [16 * x for x in harness.lengths(432, case.total_q // 16, case.max_q // 16, 3)]
        k_lengths = harness.lengths(432, case.total_kv, case.max_kv, 5)
        cu_q, cu_k = harness.cumulative(q_lengths), harness.cumulative(k_lengths)
        q = torch.randn((case.total_q, 4, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((case.total_kv, 4, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        meta = baseline._metadata(cu_q, cu_k, case.total_q, case.total_kv, 128)
        state = baseline.prepare_kv(k, v, cu_k, meta)
        q8, qs = candidate.prepare_q(q, cu_q)
        schedule = candidate.prepare_work_schedule(q_lengths, k_lengths, q.device)

        def before():
            return baseline.forward(q, k, v, cu_q, cu_k, harness.SOFTMAX_SCALE, kv_state=state, block_m=128,
                                    num_ctas=512, fast_exp=False)[:2]

        def after():
            return candidate.forward_prequantized(q8, qs, *state, cu_q, cu_k, harness.SOFTMAX_SCALE,
                                                  total_kv=case.total_kv, fast_exp=case.family == "long",
                                                  work_schedule=schedule)

        before.name, after.name = "baseline", "prepared"
        expected, expected_lse = before()
        actual, actual_lse = after()
        bitwise_equal = torch.equal(expected, actual) and torch.equal(expected_lse, actual_lse)
        if case.family == "short" and not bitwise_equal:
            raise RuntimeError(f"Unexpected exact-exp numerical change in {case.name}")
        times = harness.benchmark_interleaved([before, after], flusher, args.warmups, args.samples, args.rounds)
        key_visits = sum(
            triton.cdiv(nq, 128) * triton.cdiv(nk, 256) * 256 for nq, nk in zip(q_lengths, k_lengths) if nk)
        kv_bytes = 4 * (2 * 128 + 8) * key_visits
        throughput = {name: kv_bytes / data["timing"]["median_ms"] / 1e9 for name, data in times.items()}
        rows.append({
            "case": case.name,
            "family": case.family,
            "timings": times,
            "kv_bytes": kv_bytes,
            "effective_kv_TBps": throughput,
            "bitwise_equal": bitwise_equal,
            "speedup": times["baseline"]["timing"]["median_ms"] / times["prepared"]["timing"]["median_ms"],
        })
        print(f"{case.name}: {throughput['baseline']:.3f} -> {throughput['prepared']:.3f} effective K/V TB/s",
              flush=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        del expected, expected_lse, actual, actual_lse
        torch.cuda.empty_cache()
    report["completed"] = True
    report["families"] = {
        family: {
            "baseline_effective_kv_TBps":
            statistics.geometric_mean(r["effective_kv_TBps"]["baseline"]
                                      for r in rows
                                      if r["family"] == family),
            "prepared_effective_kv_TBps":
            statistics.geometric_mean(r["effective_kv_TBps"]["prepared"]
                                      for r in rows
                                      if r["family"] == family),
            "speedup":
            statistics.geometric_mean(r["speedup"]
                                      for r in rows
                                      if r["family"] == family),
        }
        for family in sorted({row["family"]
                              for row in rows})
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["families"], indent=2))


if __name__ == "__main__":
    main()
