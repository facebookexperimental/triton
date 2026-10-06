#!/usr/bin/env python3
"""Publish and render the tlx.ops nightly perf dashboard.

  summarize DIR --platform P --version V --sha S --out FILE
      Condense one platform's python/test/tlx_benchmark artifacts into the
      public summary uploaded to the 'tlx-ops-bench' release. Raw artifacts
      carry the runner's hostname, GPU UUID and process list, so only
      benchmark fields are copied.
  prune KEEP
      Keep the last KEEP nights of summaries on that release.
  render OUTDIR [--from DIR]
      Write OUTDIR/index.html from every summary on the release (or in DIR,
      for a local preview), for pages.yml to mount at /bench/. The page is
      tlx_ops_bench_dashboard.html with the data inlined.
"""
import argparse
import ast
import collections
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile

REPO = os.environ.get("GITHUB_REPOSITORY", "facebookexperimental/triton")
TAG = "tlx-ops-bench"
ASSET = re.compile(r"^(\d{8})\.([a-z0-9]+)\.json$")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tlx_ops_bench_dashboard.html")
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DTYPES = {
    "bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "float8_e4m3fn": "fp8e4m3", "float8_e5m2": "fp8e5m2"
}
#: One letter per status in the rendered page's per-night arrays.
STATUS_CODE = {"ok": "o", "noisy": "n", "pip": "p", "error": "e"}
#: Each op's bench script under python/test/tlx_benchmark/, which ranks as that op's harness among suspect commits.
BENCH_SCRIPT = {
    "mm": "bench_mm.py", "addmm": "bench_addmm.py", "flash_attn": "bench_flash_attn.py", "flash_attn_mxfp8":
    "bench_flash_attn_mxfp8.py", "hstu_attn_dev": "bench_hstu_attn.py", "kimi_delta_attention": "bench_kda.py",
    "kda_paged_prefill": "bench_kda_prefill.py", "kda_recurrent_decode": "bench_kda_decode.py"
}
#: Per-platform fields the page shows in its version panel, taken from the latest night.
PLATFORM_FIELDS = ("gpu", "driver", "torch", "runtime", "triton", "governed", "timing", "version", "sha")
#: A case's night-over-night TLX TFLOP/s change that makes a perf event, if the night after still holds it.
EVENT_THRESHOLD = 0.10
#: Changed cases listed per op and direction in one event; the rest are counted.
EVENT_ROWS = 50
#: Suspect commits listed per op in one event; the rest are counted.
EVENT_SUSPECTS = 6
#: Per platform: its kernels' arch tag and its compiler backend, for ranking suspect commits.
PLATFORM_TARGET = {
    "b200": ("sm100", "third_party/nvidia/"), "h100": ("sm90", "third_party/nvidia/"), "mi350":
    ("gfx950", "third_party/amd/")
}
ARCH_TAGS = ("sm90", "sm100", "gfx942", "gfx950")
#: Each op's kernel directory under third_party/tlx/ops/kernels/.
KERNEL_DIR = {
    "mm": "mm", "addmm": "addmm", "bmm": "bmm", "flash_attn": "flash_attn", "flash_attn_mxfp8": "flash_attn_mxfp8",
    "hstu_attn_dev": "hstu_attn", "kimi_delta_attention": "kda", "kda_paged_prefill": "kda", "kda_recurrent_decode":
    "kda"
}
DISPATCH = ("third_party/tlx/ops/__init__.py", "third_party/tlx/ops/_catalog.py",
            "third_party/tlx/ops/kernels/__init__.py", "third_party/tlx/language/tlx/hw/target.py")
#: A suspect's area by rank, most specific first.
AREA = {1: "kernel", 2: "shapes", 3: "dispatch", 4: "harness", 5: "compiler", 6: "compiler (common)", 7: "tlx lang"}
MEASURED = ("ok", "noisy", "pip")


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def input_params(label):
    """The `k=v` inputs of a case, parsed from the harness's `repr((args, kwargs))` label.

    dtype and direction are dropped: each is its own case field and page filter.
    """
    try:
        _, kwargs = ast.literal_eval(label)
    except (ValueError, SyntaxError, TypeError):
        return {"input": label}
    # Spaces dropped from values (a strides list) so an input reads as one token; strides last, after the sizes.
    params = {str(k): str(v).replace(" ", "") for k, v in kwargs.items() if k not in ("dtype", "dir")}
    return dict(sorted(params.items(), key=lambda kv: kv[0] == "strides"))


def summarize(args):
    ops, env = {}, {}
    # Recursive: a downloaded Actions artifact keeps the directories it was uploaded from.
    for path in sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True)):
        with open(path) as fh:
            artifact = json.load(fh)
        if not artifact.get("results"):
            continue
        env = artifact["env"]
        cases = [{
            "key": r["case"]["key"],
            "params": input_params(r["case"]["input"]),
            "direction": r["case"]["direction"],
            "dtype": DTYPES.get(r["case"]["dtype"], r["case"]["dtype"]),
            "status": r["status"],
            "speedup": r.get("speedup"),
            "tlx_tflops": (r.get("tlx") or {}).get("mean"),
            "ref_tflops": (r.get("ref") or {}).get("mean"),
            "notes": r.get("notes", []),
        } for r in artifact["results"]]
        op = artifact["results"][0]["case"]["op"]
        ops[op] = {
            "ref": env.get("ref"), "space": env.get("space"), "cold_compile": env.get("cold_compile"), "cases": cases
        }
    date = re.search(r"\.dev(\d{8})", args.version)
    gpu = env.get("gpu") or {}
    torch_version = env.get("torch") or ""
    rocm = re.search(r"\+rocm([\d.]+)", torch_version)
    summary = {
        "schema": 2,
        "platform": args.platform,
        "version": args.version,
        "sha": args.sha,
        "date": date.group(1) if date else None,
        "gpu": gpu.get("name"),
        "driver": gpu.get("driver"),
        "torch": torch_version,
        "runtime": f"CUDA {env['cuda']}" if env.get("cuda") else (f"ROCm {rocm.group(1)}" if rocm else None),
        "triton": env.get("triton"),
        "governed": (env.get("governed") or {}).get("applied", []),
        "timing": {k: env.get(k)
                   for k in ("latency_mode", "replicates")},
        "ops": ops,
    }
    with open(args.out, "w") as fh:
        json.dump(summary, fh)
    print(f"summarized {len(ops)} op(s) -> {args.out}")


def prune(args):
    try:
        release = json.loads(sh("gh", "api", f"repos/{REPO}/releases/tags/{TAG}"))
    except subprocess.CalledProcessError:
        print(f"no {TAG} release yet; nothing to prune")
        return
    by_date = collections.defaultdict(list)
    for asset in release.get("assets", []):
        m = ASSET.match(asset["name"])
        if m:
            by_date[m.group(1)].append(asset)
    stale = sorted(by_date)[:-args.keep] if len(by_date) > args.keep else []
    for date in stale:
        for asset in by_date[date]:
            sh("gh", "api", "-X", "DELETE", f"repos/{REPO}/releases/assets/{asset['id']}")
            print("pruned", asset["name"])
    print(f"kept {min(len(by_date), args.keep)} night(s); pruned {len(stale)}")


def read_summaries(directory):
    runs = collections.defaultdict(list)
    for name in sorted(os.listdir(directory)):
        if ASSET.match(name):
            with open(os.path.join(directory, name)) as fh:
                summary = json.load(fh)
            if summary.get("schema") == 2:
                runs[summary["platform"]].append(summary)
    return runs


def load_summaries():
    """Every summary on the release, by platform, oldest first. Empty if there is no release yet."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            sh("gh", "release", "download", TAG, "--repo", REPO, "--pattern", "*.json", "--dir", tmp)
        except subprocess.CalledProcessError as exc:
            print(f"no {TAG} release to render ({exc.stderr.strip()})", file=sys.stderr)
            return {}
        return read_summaries(tmp)


def commits(a, b):
    """Commits in the wheel range a..b with the files each touches; None if the checkout lacks that history."""
    try:
        out = subprocess.run(["git", "log", "--format=%x00%h\t%s", "--name-only", "--no-renames", f"{a}..{b}"],
                             capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout
    except (subprocess.CalledProcessError, OSError):
        return None
    found = []
    for chunk in out.split("\x00")[1:]:
        lines = [line for line in chunk.strip().split("\n") if line]
        sha, subject = lines[0].split("\t", 1)
        pr = re.search(r"\(#(\d+)\)\s*$", subject)
        found.append({
            "sha": sha, "title": re.sub(r"\s*\(#\d+\)\s*$", "", subject), "pr": int(pr.group(1)) if pr else None,
            "files": lines[1:]
        })
    return found


def file_rank(path, op, platform):
    """How directly `path` bears on `op` on `platform` (1 = its kernel); None if it cannot affect it."""
    arch, backend = PLATFORM_TARGET.get(platform, ("", ""))
    name = path.rsplit("/", 1)[-1]
    kernel = re.match(r"third_party/tlx/ops/kernels/([^/]+)/", path)
    if kernel:
        if kernel.group(1) != KERNEL_DIR.get(op) or any(a in name and a != arch
                                                        for a in ARCH_TAGS) and arch not in name:
            return None
        return 2 if "shape" in name else 1
    if path == "third_party/tlx/ops/kernels/_shape_suites.py":
        return 2
    if path in DISPATCH:
        return 3
    if path.startswith("python/test/tlx_benchmark/"):
        harness = path.startswith("python/test/tlx_benchmark/_harness/") or name in (BENCH_SCRIPT.get(op),
                                                                                     "conftest.py")
        return 4 if harness else None
    if path.startswith(("third_party/nvidia/", "third_party/amd/")):
        return 5 if path.startswith(backend) else None
    if path.startswith(("lib/", "include/", "python/src/", "python/triton/compiler", "python/triton/language",
                        "python/triton/runtime")):
        return 6
    if path.startswith(("third_party/tlx/language", "third_party/tlx/dialect")):
        return 7
    return None


def suspects(found, op, platform, signature):
    """Commits that could explain `op` moving, by their most relevant file; ties keep git order (newest first).

    A host-side change (same µs lost at every size) ranks dispatch and harness
    above kernels. A _catalog.py edit that only registers other ops' or
    arches' kernels is not taken as dispatch.
    """
    ranked = []
    for c in found:
        files = c["files"]
        kernels = [f for f in files if f.startswith("third_party/tlx/ops/kernels/")]
        if kernels and not any(file_rank(f, op, platform) for f in kernels):
            files = [f for f in files if f != "third_party/tlx/ops/_catalog.py"]
        ranks = [r for r in (file_rank(f, op, platform) for f in files) if r]
        if ranks:
            best = min(ranks)
            key = best - 3.5 if signature == "host" and best in (3, 4) else best
            ranked.append((key, {"sha": c["sha"], "title": c["title"], "pr": c["pr"], "area": AREA[best]}))
    return [c for _, c in sorted(ranked, key=lambda kc: kc[0])]


def numa(night):
    return next((g for g in night.get("governed") or [] if g.startswith("NUMA")), None)


def env_changes(a, b):
    """What differs between two nights' environments, as {field, from, to}."""
    changes = [{"field": label, "from": a.get(f), "to": b.get(f)}
               for f, label in (("gpu", "GPU"), ("driver", "driver"), ("torch", "torch"), ("runtime", "runtime"))
               if a.get(f) != b.get(f)]
    governed = [[g for g in n.get("governed") or [] if not g.startswith("NUMA")] for n in (a, b)]
    if governed[0] != governed[1]:
        changes.append({
            "field": "clocks", "from": " · ".join(governed[0]) or "unlocked", "to": " · ".join(governed[1])
            or "unlocked"
        })
    if numa(a) != numa(b):
        changes.append({"field": "NUMA", "from": numa(a), "to": numa(b)})
    if a.get("timing") != b.get("timing"):
        changes.append({"field": "timing", "from": json.dumps(a.get("timing")), "to": json.dumps(b.get("timing"))})
    return changes


def latency_us(case):
    """One call's time from its throughput, where the op's FLOPs are known from its inputs.

    The harness times whole calls, so this includes launch and host overhead, not only the kernel.
    """
    p = case["params"]
    if case["key"].startswith("mm/") and case.get("tlx_tflops"):
        return 2 * int(p["M"]) * int(p["N"]) * int(p["K"]) / case["tlx_tflops"] / 1e6
    return None


def signature(rows):
    """'host' when the changed cases moved by a similar number of µs, 'kernel' when by a similar %."""
    us = [r["dus"] for r in rows if r["dus"] is not None]
    if len(us) < 3:
        return None, None

    def spread(v):
        mid = statistics.median(v)
        return statistics.median(abs(x - mid) for x in v) / abs(mid) if mid else float("inf")

    return ("host" if spread(us) < spread([r["dt"] for r in rows]) else "kernel"), round(statistics.median(us), 2)


def events(nights):
    """Per night (oldest first): what changed since the night before, in the environment, the suite and perf.

    A perf event is a case whose TLX TFLOP/s moved at least EVENT_THRESHOLD
    and was still that far off the night after; on the latest night, or when
    the night after has no results for the op, that check cannot run, so its
    events are marked unconfirmed.
    """
    platform = nights[-1]["platform"]
    cases = [{name: {c["key"]: c for c in op["cases"]} for name, op in n["ops"].items()} for n in nights]
    timed = lambda c: bool(c and c["status"] in MEASURED and c.get("tlx_tflops"))
    out = []
    for i, night in enumerate(nights):
        entry = {
            "date": night["date"], "numa": numa(night), "driver": night.get("driver"), "torch": night.get("torch"),
            "env": [], "suite": {}, "perf": {}, "breaks": []
        }
        out.append(entry)
        if not i:
            continue
        before = nights[i - 1]
        found = commits(before["sha"], night["sha"])
        entry.update(prev=before["date"], range=f"{before['sha'][:9]}..{night['sha'][:9]}",
                     commits=None if found is None else len(found), env=env_changes(before, night),
                     compare=f"https://github.com/{REPO}/compare/{before['sha'][:12]}...{night['sha'][:12]}")
        for op in sorted(cases[i - 1].keys() | cases[i].keys()):
            a, b = cases[i - 1].get(op, {}), cases[i].get(op, {})
            # No next night, or one without this op at all, cannot confirm a change: it stays unconfirmed.
            nxt = (cases[i + 1].get(op) or None) if i + 1 < len(nights) else None
            both = a.keys() & b.keys()
            suite = {
                "added": len(b.keys() - a.keys()), "removed": len(a.keys() - b.keys()), "broke":
                sum(timed(a[k]) and not timed(b[k]) for k in both), "fixed":
                sum(not timed(a[k]) and timed(b[k]) for k in both), "new": not a, "gone": not b, "total": len(b)
            }
            if suite["added"] or suite["removed"] or suite["broke"] or suite["fixed"]:
                entry["suite"][op] = suite
            if not both:
                entry["breaks"].append(op)
            slower, faster = [], []
            for k in both:
                x, y = a[k], b[k]
                z = nxt.get(k) if nxt is not None else None
                if not (timed(x) and timed(y)) or nxt is not None and not timed(z):
                    continue
                dt = y["tlx_tflops"] / x["tlx_tflops"] - 1
                held = None if z is None else z["tlx_tflops"] / x["tlx_tflops"] - 1
                lx, ly = latency_us(x), latency_us(y)
                row = {
                    "key": k.split("/", 2)[-1], "base": round(x["tlx_tflops"], 2), "now": round(y["tlx_tflops"], 2),
                    "next": None if z is None else round(z["tlx_tflops"], 2), "dt": round(dt, 4), "us":
                    None if lx is None else round(lx, 1), "us_now": None if ly is None else round(ly, 1), "dus":
                    None if lx is None else round(ly - lx, 2)
                }
                if dt <= -EVENT_THRESHOLD and (held is None or held <= -EVENT_THRESHOLD):
                    slower.append(row)
                elif dt >= EVENT_THRESHOLD and (held is None or held >= EVENT_THRESHOLD):
                    faster.append(row)
            if not (slower or faster):
                continue
            slower.sort(key=lambda r: r["dt"])
            faster.sort(key=lambda r: -r["dt"])
            sig, median_dus = signature(slower if len(slower) >= len(faster) else faster)
            ranked = suspects(found or [], op, platform, sig)
            entry["perf"][op] = {
                "slower": slower[:EVENT_ROWS], "faster": faster[:EVENT_ROWS], "n_slower": len(slower), "n_faster":
                len(faster), "signature": sig, "median_dus": median_dus, "unconfirmed": nxt is None, "compared":
                sum(timed(a[k]) and timed(b[k])
                    for k in both), "suspects": ranked[:EVENT_SUSPECTS], "more": max(0,
                                                                                     len(ranked) - EVENT_SUSPECTS)
            }
    return out


def page_data(runs):
    """The page's data, compacted: each case's description once per op, then per night only the numbers.

    mm alone is ~400 cases a night; repeating each case's inputs for every
    kept night would make the page megabytes of the same strings.
    """
    platforms = {}
    for platform, nights in runs.items():
        latest = nights[-1]
        catalog = {}  # op -> {case key: index}
        ops = {}
        for night in nights:
            for name, op in night["ops"].items():
                entry = ops.setdefault(name, {"cases": []})
                entry.update(ref=op.get("ref"), space=op.get("space"), cold=op.get("cold_compile"))
                index = catalog.setdefault(name, {})
                for c in op["cases"]:
                    if c["key"] not in index:
                        index[c["key"]] = len(entry["cases"])
                        entry["cases"].append([c["params"], c["direction"], c["dtype"]])
        encoded = []
        for night in nights:
            runs_by_op = {}
            for name, op in night["ops"].items():
                row = [None] * len(ops[name]["cases"])
                for c in op["cases"]:
                    cell = [STATUS_CODE.get(c["status"], "e"), c["speedup"], c["tlx_tflops"], c["ref_tflops"]]
                    if night is latest:  # notes are only shown for the latest night
                        cell.append("; ".join(c["notes"]))
                    row[catalog[name][c["key"]]] = [round(v, 4) if isinstance(v, float) else v for v in cell]
                runs_by_op[name] = row
            encoded.append({"date": night["date"], "ops": runs_by_op})
        meta = {k: latest.get(k) for k in PLATFORM_FIELDS}
        latest_ops = {name: ops[name] for name in latest["ops"]}
        platforms[platform] = {**meta, "ops": latest_ops, "nights": encoded, "events": events(nights)}
    return {"repo": REPO, "tag": TAG, "threshold": EVENT_THRESHOLD, "platforms": platforms}


def render(args):
    runs = read_summaries(args.source) if args.source else load_summaries()
    data = json.dumps(page_data(runs), separators=(",", ":")).replace("</", "<\\/")
    with open(TEMPLATE) as fh:
        page = fh.read()
    if page.count("/*__DATA__*/null") != 1:
        raise SystemExit(f"{TEMPLATE} must contain the /*__DATA__*/null placeholder exactly once")
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "index.html")
    with open(out, "w") as fh:
        fh.write(page.replace("/*__DATA__*/null", data))
    print(f"rendered {sum(len(n) for n in runs.values())} run(s), {len(data) // 1024} KB of data -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    for flag in ("--platform", "--version", "--sha", "--out"):
        s.add_argument(flag, required=True)
    p = sub.add_parser("prune")
    p.add_argument("keep", type=int)
    r = sub.add_parser("render")
    r.add_argument("outdir")
    r.add_argument("--from", dest="source", help="read summaries from this directory instead of the release")
    args = parser.parse_args()
    {"summarize": summarize, "prune": prune, "render": render}[args.cmd](args)


if __name__ == "__main__":
    main()
