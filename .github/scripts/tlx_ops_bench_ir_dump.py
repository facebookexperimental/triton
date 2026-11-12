#!/usr/bin/env python3
"""Compile tlx.ops benchmark cases with the installed Triton and dump every kernel's IR, PTX/AMDGCN and SASS.

Run once per wheel with the same harness, then compare the two dumps with tlx_ops_bench_ir_diff.py. Each compile is
recorded in index.json under a key built from the kernel name, signature, constexprs and attributes, so the same
specialization can be matched across compiler versions even when its dump directory hash differs.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback


def spec_key(src):
    get = lambda v: v.cache_key if hasattr(v, "cache_key") else str(v)
    signature = [(str(k), str(v)) for k, v in sorted(src.signature.items(), key=lambda kv: str(kv[0]))]
    constants = [(str(k), get(v)) for k, v in sorted(src.constants.items(), key=lambda kv: str(kv[0]))]
    return json.dumps([src.name, signature, constants, str(src.attrs)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, help="python/test/tlx_benchmark directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--module", action="append", help="bench module(s) to dump; default all except torchtlx")
    parser.add_argument("--key", help="dump only this case key")
    parser.add_argument("--direction", help="dump only this direction")
    parser.add_argument("--gpu-uuid", help="pin this GPU")
    parser.add_argument("--corpus-out", type=Path,
                        help="also write the compiled specializations as a tlx_codegen_snapshot.py --corpus file")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # A private, empty cache makes every kernel compile, and so dump, once. TRITON_ALWAYS_COMPILE would do the same
    # but also disables the launcher fast path, which some kernels' launches rely on.
    out = args.out.resolve()
    os.environ.update(TRITON_KERNEL_DUMP="1", TRITON_DUMP_DIR=str(out / "dump"), TRITON_CACHE_DIR=str(out / "cache"))
    sys.path.insert(0, str(Path(args.harness).resolve()))

    import torch
    import triton
    import triton.compiler as compiler
    from triton.runtime.cache import get_dump_manager

    compiles, current, corpus, unreplayable = [], {"case": None}, {}, set()
    original = compiler.compile
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tlx_codegen_snapshot import corpus_entry

    def recording_compile(src, *a, **k):
        if hasattr(src, "constants"):
            dump = Path(get_dump_manager(src.hash()).cache_dir).name
            compiles.append({"key": spec_key(src), "name": src.name, "dump": dump, "case": current["case"]})
            if args.corpus_out:
                entry = corpus_entry(src, k.get("options"), k.get("target"))
                if entry is None:
                    unreplayable.add(src.name)
                else:
                    corpus.setdefault(entry["id"], {**entry, "cases": []})["cases"].append(current["case"])
        return original(src, *a, **k)

    # JITFunction imports `compile` from triton.compiler when it first builds its device cache.
    compiler.compile = recording_compile
    from _harness import driver
    from _harness.denoise import list_devices, select_device
    device = (next(
        (d for d in list_devices() if d.uuid == args.gpu_uuid), None) if args.gpu_uuid else select_device("auto"))
    if device is None:
        raise SystemExit(f"GPU {args.gpu_uuid} not found")
    driver.select(device)
    device.pin()

    names = args.module or sorted(p.stem for p in Path(args.harness).glob("bench_*.py") if "torchtlx" not in p.stem)
    cases = []
    for name in names:
        try:
            bench = importlib.import_module(name)
            if not bench.supported():
                continue
            # Ops that autotune a full space compile every config; the smoke space keeps the dump to a few.
            space = driver.resolve_space(bench, None)
            space = "smoke" if space == "full" else space
            for case in bench.cases(False, None):
                if (args.key and case.key != args.key) or (args.direction and case.direction != args.direction):
                    continue
                current["case"] = f"{case.key}|{case.direction}"
                before = len(compiles)
                try:
                    prep = bench.prepare(case, space)
                    prep.tlx_fn()
                    torch.cuda.synchronize()
                    status = "ok"
                    del prep
                except Exception as exc:
                    status = f"error: {type(exc).__name__}: {str(exc)[:300]}"
                torch.cuda.empty_cache()
                cases.append(
                    {"module": name, "case": current["case"], "status": status, "compiles": len(compiles) - before})
        except Exception:
            cases.append({"module": name, "case": None, "status": "error: " + traceback.format_exc(limit=2)[-300:]})
    index = {
        "triton": triton.__version__, "device": device.name, "gpu_uuid": device.uuid, "cases": cases, "compiles":
        compiles
    }
    (args.out / "index.json").write_text(json.dumps(index, indent=1))
    if args.corpus_out:
        entries = sorted(corpus.values(), key=lambda e: (e["kernel"], e["id"]))
        # One entry per line keeps the checked-in corpus reviewable entry by entry.
        head = json.dumps({
            "triton": triton.__version__, "device": device.name, "capability": list(torch.cuda.get_device_capability())
        })[:-1]
        args.corpus_out.write_text(head + ', "entries": [\n' + ",\n".join(json.dumps(e) for e in entries) + "\n]}\n")
        print(f"corpus: {len(entries)} specializations; not replayable from source: {sorted(unreplayable) or 'none'}")
    print(f"{triton.__version__}: {len(cases)} cases, {len(compiles)} compiles, "
          f"{sum(c['status'] != 'ok' for c in cases)} case errors")


if __name__ == "__main__":
    main()
