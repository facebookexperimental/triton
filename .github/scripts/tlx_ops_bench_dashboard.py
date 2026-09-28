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
import subprocess
import sys
import tempfile

REPO = os.environ.get("GITHUB_REPOSITORY", "facebookexperimental/triton")
TAG = "tlx-ops-bench"
ASSET = re.compile(r"^(\d{8})\.([a-z0-9]+)\.json$")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tlx_ops_bench_dashboard.html")
DTYPES = {
    "bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "float8_e4m3fn": "fp8e4m3", "float8_e5m2": "fp8e5m2"
}
#: One letter per status in the rendered page's per-night arrays.
STATUS_CODE = {"ok": "o", "noisy": "n", "pip": "p", "error": "e"}
#: Per-platform fields the page shows in its version panel, taken from the latest night.
PLATFORM_FIELDS = ("gpu", "driver", "torch", "runtime", "triton", "governed", "timing", "version", "sha")


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
        platforms[platform] = {**meta, "ops": latest_ops, "nights": encoded}
    return {"repo": REPO, "tag": TAG, "platforms": platforms}


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
