#!/usr/bin/env python3
"""Same-runner A/B of the tlx.ops perf smoke subset, for bundle PRs.

  bench    On the GPU runner. Installs the base and head wheels in turn and runs
           the smoke subset with each, `--rounds` times, alternating which goes
           first, all on one GPU. Writes RESULTS (results.json in --out): the
           benchmark fields of every run plus the GPU, driver and software
           versions, without the host name, GPU UUID or process list that the
           raw harness artifacts carry.
  report   Anywhere. Compares head with base round by round, adds the nightly
           baseline as context, and writes the platform's markdown section and
           a JSON verdict.
  relevant Anywhere. Prints whether BASE..HEAD changes anything outside tests,
           CI and docs, so a head that cannot move performance is not re-run.

Why same-runner: a nightly baseline comes from another day and often another
runner, so main's own changes and 5-10% runner-to-runner noise sit inside its
ratio. Base and head measured back to back on one GPU cancel both.

A case is flagged when the median of its per-round head/base throughput ratios
moves by at least --threshold and every round moves the same way.
"""
import argparse
import base64
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

SCHEMA = 1
SIDES = ("base", "head")
#: Where the harness writes one artifact per op (python/test/tlx_benchmark/_harness/driver.py).
HARNESS_OUT = Path("/tmp/tlx_benchmark")
HARNESS_TEST = "python/test/tlx_benchmark/test_ops_perf.py"
SMOKE_SELECT = "bench_mm or bench_flash_attn or bench_norm"
SMOKE_HEAD = 8
LIBTRITON = "triton/_C/libtriton.so"
NIGHTLY_NAME = re.compile(r"(\d{8})\.([a-z0-9]+)\.json$")
#: Statuses whose throughput is a measurement (the harness's `noisy` still has a number, flagged as such).
MEASURED = ("ok", "pip", "noisy")
#: Per-case rows in the comment; GitHub caps a comment at 65536 characters and every platform shares one.
MAX_TABLE_ROWS = 60

# --------------------------------------------------------------------------------------------------------------------
# bench


def wheel_lib_digest(wheel):
    """The RECORD digest of libtriton.so in WHEEL, e.g. 'sha256=...'."""
    with zipfile.ZipFile(wheel) as zf:
        record = next(n for n in zf.namelist() if n.endswith(".dist-info/RECORD"))
        rows = csv.reader(io.StringIO(zf.read(record).decode()))
        for row in rows:
            if row and row[0] == LIBTRITON:
                return row[1]
    raise ValueError(f"{wheel} has no {LIBTRITON} in its RECORD")


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256=" + base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()


def installed_lib(python):
    """Path of the installed libtriton.so, found without importing triton."""
    code = ("import importlib.util, os; spec = importlib.util.find_spec('triton'); "
            "print(os.path.join(list(spec.submodule_search_locations)[0], '_C', 'libtriton.so'))")
    return subprocess.run([python, "-c", code], check=True, capture_output=True, text=True).stdout.strip()


def install(python, wheel):
    """Install WHEEL over the current triton and prove the files on disk are WHEEL's."""
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "pip", "install", "--python", python, "--no-deps", "--force-reinstall", "--quiet", str(wheel)]
    else:
        cmd = [python, "-m", "pip", "install", "--no-deps", "--force-reinstall", "--quiet", str(wheel)]
    subprocess.run(cmd, check=True)
    lib = installed_lib(python)
    if file_digest(lib) != wheel_lib_digest(wheel):
        raise RuntimeError(f"{lib} does not match {Path(wheel).name}")


def describe_gpu(python, repo, spec):
    """The GPU the harness picks for SPEC ('auto' or a physical index), as a dict, or None.

    Every run is then pinned to that one GPU. The dict carries the name and
    arch the report shows, because the harness records neither on AMD. The
    harness package imports triton, so this only works once a wheel is installed.
    """
    code = ("import json, sys; sys.path.insert(0, 'python/test/tlx_benchmark'); "
            "from _harness.denoise import select_device; "
            f"d = select_device({spec!r}); "
            "print(json.dumps({'index': d.index, 'name': d.name, 'vendor': d.vendor, 'arch': d.arch}))")
    try:
        out = subprocess.run([python, "-c", code], cwd=repo, check=True, capture_output=True, text=True, timeout=120)
        device = json.loads(out.stdout.strip().splitlines()[-1])
        int(device["index"])
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip().splitlines()[-1:]
        print(f"could not identify the GPU ({tail[0] if tail else exc}); each run picks its own", file=sys.stderr)
        return None
    except (subprocess.SubprocessError, ValueError, IndexError, KeyError, TypeError) as exc:
        print(f"could not identify the GPU ({exc}); each run picks its own", file=sys.stderr)
        return None
    if device.get("vendor") == "amd":
        device["driver"] = amd_driver_version()
    return device


def amd_driver_version():
    """The amdgpu kernel driver version, e.g. '6.14.14', or None."""
    try:
        return Path("/sys/module/amdgpu/version").read_text().strip() or None
    except OSError:
        return None


def sanitize_env(env):
    gpu = env.get("gpu") or {}
    governed = env.get("governed")
    return {
        "gpu": gpu.get("name"),
        "driver": gpu.get("driver"),
        "cuda": env.get("cuda"),
        "torch": env.get("torch"),
        "triton": env.get("triton"),
        "governed": list(governed.get("applied") or []) if isinstance(governed, dict) else None,
    }


def sanitize_results(directory):
    """(cases, env, clocks) from one run's harness artifacts: benchmark fields only."""
    cases, env, clocks = [], {}, {}
    for path in sorted(Path(directory).glob("*.json")):
        try:
            artifact = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not artifact.get("results"):
            continue
        env = env or sanitize_env(artifact.get("env") or {})
        trace = ((artifact.get("env") or {}).get("run") or {}).get("clock_trace") or {}
        if trace:
            clocks[path.stem] = {"median_mhz": trace.get("median_mhz"), "stable": trace.get("stable")}
        for result in artifact["results"]:
            case, tlx = result.get("case") or {}, result.get("tlx") or {}
            cases.append({
                "key": case.get("key"),
                "op": case.get("op"),
                "arch": case.get("arch"),
                "direction": case.get("direction"),
                "status": result.get("status"),
                "tflops": tlx.get("mean"),
                "cv": tlx.get("cv"),
                "flop_count": result.get("flop_count"),
                "notes": [str(n)[:300] for n in (result.get("notes") or [])][:3],
            })
    return cases, env, clocks


def run_side(args, side, rnd, wheel, gpu, out, installed=False):
    """Install WHEEL (unless INSTALLED) and run the smoke subset once. Returns (run record, sanitized env)."""
    record = {"side": side, "round": rnd, "rc": None, "seconds": None, "error": None, "cases": []}
    raw = out / "raw" / side / f"round{rnd}"
    raw.mkdir(parents=True, exist_ok=True)
    try:
        if not installed:
            install(sys.executable, wheel)
    except (subprocess.SubprocessError, OSError, RuntimeError, ValueError, StopIteration) as exc:
        record["error"] = f"install failed: {exc}"
        return record, {}
    shutil.rmtree(HARNESS_OUT, ignore_errors=True)
    env = os.environ.copy()
    # Each side keeps its own kernel and autotune cache across rounds; nothing is shared between the two builds.
    env["TRITON_HOME"] = str((out / "triton-home" / side).resolve())
    cmd = [
        sys.executable, "-m", "pytest", HARNESS_TEST, "--import-mode=importlib", "-p", "no:logging", "-p",
        "no:cacheprovider", "-k", args.select, "--fwd-only", "--head",
        str(args.head_cases), "--cold-compile", "none", "--gpu", gpu, "-v", "-s", f"--timeout={args.case_timeout}",
        f"--junitxml={raw / 'junit.xml'}"
    ]
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=args.repo, env=env, timeout=args.run_timeout)
        record["rc"] = proc.returncode
    except subprocess.TimeoutExpired:
        record["error"] = f"timed out after {args.run_timeout}s"
    record["seconds"] = round(time.monotonic() - start, 1)
    for path in HARNESS_OUT.glob("*.json"):
        shutil.copy2(path, raw / path.name)
    record["cases"], run_env, record["clocks"] = sanitize_results(raw)
    if not record["cases"] and not record["error"]:
        record["error"] = f"no results (pytest exit {record['rc']})"
    return record, run_env


def bench(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wheels = {side: getattr(args, f"{side}_wheel") for side in SIDES}
    wheels = {side: Path(w) if w else None for side, w in wheels.items()}
    gpu = args.gpu
    results = {
        "schema": SCHEMA, "platform": args.platform, "rounds": args.rounds, "gpu_index": gpu, "select": args.select,
        "head_cases": args.head_cases, "wheels": {side: w.name if w else None
                                                  for side, w in wheels.items()}, "env": {}, "device": None, "runs": []
    }
    sides = [side for side in SIDES if wheels[side]]
    for rnd in range(1, args.rounds + 1):
        # Alternate who goes first, so a drift during the job (clocks, thermals) lands on both sides equally.
        order = sides if rnd % 2 else sides[::-1]
        for side in order:
            print(f"== round {rnd}/{args.rounds}: {side} ({wheels[side].name})", flush=True)
            installed = False
            if not results["runs"]:
                # Identify and pin the GPU before the first run; the harness's picker needs an installed triton.
                try:
                    install(sys.executable, wheels[side])
                    installed = True
                except (subprocess.SubprocessError, OSError, RuntimeError, ValueError, StopIteration):
                    pass
                device = describe_gpu(sys.executable, args.repo, gpu)
                if device:
                    results["device"] = device
                    gpu = results["gpu_index"] = str(device["index"])
            record, env = run_side(args, side, rnd, wheels[side], gpu, out, installed)
            results["env"] = results["env"] or env
            results["runs"].append(record)
            print(
                f"== {side} round {rnd} on gpu {gpu}: {len(record['cases'])} cases in {record.get('seconds')}s"
                f"{'; ' + record['error'] if record.get('error') else ''}", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "results.json").write_text(json.dumps(results, indent=1))
    return 0


# --------------------------------------------------------------------------------------------------------------------
# report


def measured(case):
    value = case.get("tflops")
    return case.get("status") in MEASURED and isinstance(value, (int, float)) and value > 0


def by_round(results):
    """{side: {round: {key: case}}}; a round that failed to install or run has no entry."""
    runs = {side: {} for side in SIDES}
    for run in results.get("runs", []):
        if run.get("cases"):
            runs[run["side"]][run["round"]] = {c["key"]: c for c in run["cases"] if c.get("key")}
    return runs


def latency_us(case):
    """Kernel time in microseconds, from the harness's FLOP count and throughput."""
    flops, tflops = case.get("flop_count"), case.get("tflops")
    if not measured(case) or not isinstance(flops, (int, float)) or flops <= 0:
        return None
    return flops / tflops / 1e6


def median_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def compare(results, threshold):
    """One row per case key seen on either side."""
    runs = by_round(results)
    keys = sorted({k for side in SIDES for rnd in runs[side].values() for k in rnd})
    rows = []
    for key in keys:
        seen = {side: [rnd[key] for rnd in runs[side].values() if key in rnd] for side in SIDES}
        row = {"key": key, "group": "/".join(key.split("/")[:2]), "ratio": None, "ratio_min": None, "ratio_max": None}
        row["pairs"], row["notes"], row["nightly_ratio"] = 0, [], None
        for side in SIDES:
            row[f"{side}_us"] = median_or_none(latency_us(c) for c in seen[side])
            row[f"{side}_tflops"] = median_or_none(c["tflops"] for c in seen[side] if measured(c))
        row["noisy"] = any(c.get("status") == "noisy" for side in SIDES for c in seen[side])
        if not seen["base"] or not seen["head"]:
            row["state"] = "only-head" if seen["head"] else "only-base"
            rows.append(row)
            continue
        errors = {side: [c for c in seen[side] if c.get("status") == "error"] for side in SIDES}
        if errors["head"] and not errors["base"]:
            row["state"] = "head-error"
            row["notes"] = errors["head"][0].get("notes") or []
            rows.append(row)
            continue
        ratios = []
        for rnd in sorted(set(runs["base"]) & set(runs["head"])):
            base, head = runs["base"][rnd].get(key), runs["head"][rnd].get(key)
            if base and head and measured(base) and measured(head):
                ratios.append(head["tflops"] / base["tflops"])
        row["pairs"] = len(ratios)
        if len(ratios) < 2:
            row["state"] = "base-error" if errors["base"] and not errors["head"] else "unmeasured"
            if errors["base"]:
                row["notes"] = errors["base"][0].get("notes") or []
            rows.append(row)
            continue
        row["ratio"], row["ratio_min"], row["ratio_max"] = statistics.median(ratios), min(ratios), max(ratios)
        if row["ratio"] <= 1 - threshold and row["ratio_max"] < 1:
            row["state"] = "slower"
        elif row["ratio"] >= 1 + threshold and row["ratio_min"] > 1:
            row["state"] = "faster"
        else:
            row["state"] = "same"
        rows.append(row)
    return rows


def geomean(values):
    values = [v for v in values if v]
    return math.exp(statistics.fmean(math.log(v) for v in values)) if values else None


def summarize_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    out = []
    for name, members in sorted(groups.items()):
        ratios = [r["ratio"] for r in members if r["ratio"] is not None]
        out.append({
            "group": name,
            "cases": len(ratios),
            "geomean": geomean(ratios),
            "worst": min(ratios) if ratios else None,
            "best": max(ratios) if ratios else None,
            "slower": sum(r["state"] == "slower" for r in members),
            "faster": sum(r["state"] == "faster" for r in members),
            "problems": sum(r["state"] in ("head-error", "base-error", "unmeasured") for r in members),
        })
    return out


def pick_baseline(history_dir, base_date, platform):
    """Newest nightly summary dated on or before BASE_DATE (else the oldest), as (date, path) or (None, None)."""
    found = []
    for path in Path(history_dir).glob(f"*.{platform}.json") if history_dir else []:
        match = NIGHTLY_NAME.search(path.name)
        if match and match.group(2) == platform:
            found.append((match.group(1), path))
    if not found:
        return None, None
    found.sort()
    before = [f for f in found if f[0] <= base_date]
    return before[-1] if before else found[0]


def load_nightly(path):
    cases = {}
    for op in (json.loads(Path(path).read_text()).get("ops") or {}).values():
        for case in op.get("cases") or []:
            value = case.get("tlx_tflops")
            if isinstance(value, (int, float)) and value > 0 and case.get("status") in MEASURED:
                cases[case.get("key")] = value
    return cases


def add_nightly(rows, nightly, has_ab):
    """Fill each row's head-vs-nightly ratio; return (side, median ratio) for the runner context line.

    The context line uses base when there is one: base is the main commit the nightly measured, give or take a day,
    so its ratio is this runner's own difference from the nightly runner and is not moved by the PR.
    """
    for row in rows:
        reference = nightly.get(row["key"])
        if reference and row["head_tflops"]:
            row["nightly_ratio"] = row["head_tflops"] / reference
    side = "base" if has_ab else "head"
    return side, median_or_none(row[f"{side}_tflops"] / nightly[row["key"]]
                                for row in rows
                                if row[f"{side}_tflops"] and nightly.get(row["key"]))


def verdict_for(rows, has_ab):
    if not any(r["state"] != "only-base" for r in rows):
        return "NO SIGNAL"
    if has_ab:
        if any(r["state"] in ("slower", "head-error") for r in rows):
            return "INVESTIGATE"
        return "PASS"
    # Head only: the old nightly comparison, with its looser thresholds for cross-day, cross-runner noise.
    ratios = [r["nightly_ratio"] for r in rows if r["nightly_ratio"] is not None]
    if not ratios:
        return "NO SIGNAL"
    return "INVESTIGATE" if sum(r < 0.80 for r in ratios) >= 2 or any(r < 0.60 for r in ratios) else "PASS"


DISPLAY = {"b200": "B200", "h100": "H100", "mi350": "MI350"}
#: What a kernel that could not be compared shows instead of a change.
PROBLEMS = {
    "head-error": "fails on this PR",
    "base-error": "fails on base",
    "unmeasured": "not measured",
    "only-head": "new on this PR",
}


def display_name(platform):
    return DISPLAY.get(platform, platform.upper())


def op_name(group):
    """'flash_attn_varlen/sm90' -> 'flash_attn_varlen': each section already names its GPU."""
    return group.split("/")[0]


def plural(count, word):
    return f"{count} {word}{'' if count == 1 else 's'}"


def change_pct(ratio):
    """Throughput change in percent: a head/base ratio of 0.81 is -19."""
    return (ratio - 1) * 100


def change_text(ratio):
    """'19% slower', '6% faster' or 'no change' for a head/base throughput ratio."""
    if ratio is None:
        return "-"
    change = change_pct(ratio)
    if round(abs(change)) == 0:
        return "no change"
    return f"{abs(change):.0f}% {'faster' if change > 0 else 'slower'}"


def signed_pct(ratio):
    """'+3%', '−8%' or '0%' for a head/base throughput ratio."""
    value = round(change_pct(ratio))
    return "0%" if value == 0 else f"{'+' if value > 0 else '−'}{abs(value)}%"


def range_text(low, high):
    """The runs' spread for a table cell: '−8% to +8%', or one value when they round alike."""
    low_text, high_text = signed_pct(low), signed_pct(high)
    return low_text if low_text == high_text else f"{low_text} to {high_text}"


def range_words(low, high):
    """The runs' spread in words: 'from 8% slower to 8% faster'."""
    return f"from {change_text(low)} to {change_text(high)}"


def by_how_much(ratios):
    """'by 7%' or 'by 19–26%' for flagged ratios that all moved one way."""
    amounts = sorted(round(abs(change_pct(r))) for r in ratios)
    return f"by {amounts[0]}%" if amounts[0] == amounts[-1] else f"by {amounts[0]}–{amounts[-1]}%"


def within(ratios):
    """'within ±3%' for unflagged ratios."""
    spread = max((abs(change_pct(r)) for r in ratios), default=0)
    return f"within ±{max(1, math.ceil(spread - 1e-9))}%"  # 1e-9: (0.99 - 1) * 100 is 1.0000000000000009


def fmt_us(value):
    if value is None:
        return "-"
    return f"{value:.1f} µs" if value < 1000 else f"{value:,.0f} µs"


def env_line(env, device=None):
    """GPU, driver, runtime, torch and clock state; DEVICE fills what the harness left out (AMD)."""
    device = device or {}
    name = env.get("gpu")
    if not name and device.get("name"):
        vendor = "AMD " if device.get("vendor") == "amd" else ""
        arch = f" ({device['arch']})" if device.get("arch") else ""
        name = f"{vendor}{device['name']}{arch}"
    parts = [name]
    driver = env.get("driver") or device.get("driver")
    if driver:
        parts.append(f"driver {driver}")
    torch_version = env.get("torch") or ""
    rocm = re.search(r"\+rocm([\d.]+)", torch_version)
    if rocm:
        parts.append(f"ROCm {rocm.group(1)}")
    elif env.get("cuda"):
        parts.append(f"CUDA {env['cuda']}")
    if torch_version:
        parts.append(f"torch {torch_version}")
    governed = env.get("governed")
    if governed is not None:
        # The harness's denoise steps: "clock lock N MHz" on NVIDIA, "perf determinism N MHz" on AMD.
        lock = next((re.search(r"(\d+) MHz", g) for g in governed if g.startswith(("clock lock", "perf determinism"))),
                    None)
        parts.append(f"clock locked at {lock.group(1)} MHz" if lock else "clock not locked (noisier)")
    return ", ".join(p for p in parts if p)


def noise_line(rows):
    """How far the runs of one kernel disagree on this GPU: the noise the threshold sits on."""
    spreads = sorted(((r["ratio_max"] - r["ratio_min"]) / r["ratio"], r["key"])
                     for r in rows
                     if r["ratio"] is not None and r["pairs"] >= 2)
    if not spreads:
        return None
    median = statistics.median(value for value, _ in spreads)
    widest, key = spreads[-1]
    return (f"**Noise on this GPU:** the runs of one kernel typically differ by {median:.1%}; "
            f"the most was {widest:.0%} (`{key}`).")


def headline(rows, verdict, has_ab):
    """The section title's verdict, in words."""
    if verdict == "NO SIGNAL":
        return "⚪", "no result"
    if not has_ab:
        if verdict == "INVESTIGATE":
            return "⚠️", "kernels far below the nightly (no base to compare)"
        return "✅", "no large slowdowns vs the nightly (no base to compare)"
    count = {state: sum(r["state"] == state for r in rows) for state in ("slower", "head-error", "faster")}
    parts = []
    if count["slower"]:
        parts.append(f"{plural(count['slower'], 'kernel')} slower")
    if count["head-error"]:
        parts.append(f"{plural(count['head-error'], 'kernel')} fail on this PR")
    if parts:
        return ("❌" if count["head-error"] else "⚠️"), ", ".join(parts)
    if count["faster"]:
        return "✅", f"no slowdowns, {plural(count['faster'], 'kernel')} faster"
    return "✅", "no slowdowns found"


def findings(rows, threshold):
    """Plain bullets: what moved or broke, per op, then how close the rest stayed."""
    lines = []
    for group in sorted({r["group"] for r in rows}):
        members = [r for r in rows if r["group"] == group and r["state"] != "only-base"]
        name = f"**{op_name(group)}**"
        for state, words in (("head-error", "fail on this PR but run on base"), ("slower", "slower"),
                             ("faster", "faster"), ("base-error", "fail on base but run on this PR"),
                             ("unmeasured", "could not be measured on both sides")):
            hits = [r for r in members if r["state"] == state]
            if not hits:
                continue
            line = f"- {name}: {len(hits)} of {plural(len(members), 'kernel')} {words}"
            if state in ("slower", "faster"):
                line += f", {by_how_much([r['ratio'] for r in hits])}"
                if len(hits) <= 2:
                    line += " (" + ", ".join(f"`{r['key']}`" for r in hits) + ")"
                line += ". Every run agreed."
            else:
                note = next((r["notes"][0] for r in hits if r["notes"]), "")
                line += f": `{shorten(note)}`" if note else "."
            lines.append(line)
    same = [r for r in rows if r["state"] == "same"]
    if same:
        # Moved past the threshold but the runs disagreed: noise, and worth saying so.
        noisy = [r for r in same if abs(change_pct(r["ratio"])) >= threshold * 100]
        quiet = [r["ratio"] for r in same if r not in noisy]
        who = f"The other {len(same)} kernels" if lines else f"All {len(same)} kernels"
        line = f"{who} are {within(quiet)} of base"
        if noisy:
            line += ", except " + ", ".join(f"`{r['key']}` ({change_text(r['ratio'])}, but its runs ranged "
                                            f"{range_words(r['ratio_min'], r['ratio_max'])}, so that is noise)"
                                            for r in noisy)
        lines.append(("- " if lines else "") + line + ".")
    return lines


def group_result(members):
    """One table cell: what happened to an op's kernels."""
    parts = []
    failing = [r for r in members if r["state"] == "head-error"]
    if failing:
        parts.append(f"❌ {len(failing)} fail on this PR")
    for state in ("slower", "faster"):
        hits = [r for r in members if r["state"] == state]
        if hits:
            icon = "⚠️" if state == "slower" else "✅"
            parts.append(f"{icon} {len(hits)} {state}, {by_how_much([r['ratio'] for r in hits])}")
    other = [r for r in members if r["state"] in ("base-error", "unmeasured")]
    if other:
        parts.append(f"{len(other)} not compared")
    same = [r["ratio"] for r in members if r["state"] == "same"]
    if same:
        parts.append(f"{'✅ ' if not parts else ''}{'the rest ' if parts else ''}{within(same)}")
    return "; ".join(parts) or "-"


def nightly_age(base_date, nightly_date):
    """'; it is N days older than the base commit' when the nightly is more than a day off, else ''."""
    try:
        days = (datetime.date.fromisoformat(f"{base_date[:4]}-{base_date[4:6]}-{base_date[6:8]}") -
                datetime.date.fromisoformat(f"{nightly_date[:4]}-{nightly_date[4:6]}-{nightly_date[6:8]}")).days
    except (TypeError, ValueError):
        return ""
    if days > 1:
        return f"; it is {days} days older than the base commit"
    if days < 0:
        return f"; it is {-days} day{'s' if days < -1 else ''} newer than the base commit"
    return ""


def shorten(text, limit=300):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def kernel_table(rows, has_ab):
    """The collapsed per-kernel table, slowest and broken first."""
    table = sorted((r for r in rows if r["state"] != "only-base"), key=lambda r:
                   (r["ratio"] is not None, r["ratio"] if r["ratio"] is not None else 0, r["key"]))
    if not table:
        return []
    shown = table[:MAX_TABLE_ROWS]
    if has_ab:
        lines = [
            f"<details><summary>All {len(table)} kernels, slowest first</summary>", "",
            "| Kernel | Base | This PR | Change | Range over runs |", "|---|---:|---:|---|---|"
        ]
        for r in shown:
            if r["ratio"] is None:
                change, spread = f"{'❌ ' if r['state'] == 'head-error' else ''}{PROBLEMS.get(r['state'], r['state'])}", ""
            else:
                icon = {"slower": "⚠️ ", "faster": "✅ "}.get(r["state"], "")
                change, spread = icon + change_text(r["ratio"]), range_text(r["ratio_min"], r["ratio_max"])
            lines.append(f"| `{r['key']}` | {fmt_us(r['base_us'])} | {fmt_us(r['head_us'])} | {change} | {spread} |")
    else:
        lines = [
            f"<details><summary>All {len(table)} kernels, slowest first</summary>", "",
            "| Kernel | This PR | vs nightly |", "|---|---:|---|"
        ]
        for r in sorted(table, key=lambda r:
                        (r["nightly_ratio"] is not None, r["nightly_ratio"] or 0, r["key"]))[:MAX_TABLE_ROWS]:
            lines.append(f"| `{r['key']}` | {fmt_us(r['head_us'])} | {change_text(r['nightly_ratio'])} |")
    if len(table) > len(shown):
        lines.append(f"\n{len(table) - len(shown)} more in the run's `results.json` artifact.")
    return lines + ["", "</details>"]


def method_lines(args, results, rows, has_ab, baseline_date, context, merged):
    """The collapsed 'how this was measured' notes."""
    rounds = max([r["round"] for r in results.get("runs", [])] or [0])
    gpu = env_line(results.get("env") or {}, results.get("device"))
    lines = [f"- **GPU:** {gpu}." if gpu else "- **GPU:** not recorded."]
    pr = f"this PR `{args.head[:12]}`"
    if merged:
        pr += f" (GitHub's merge of PR head `{args.pr_head[:12]}` into base)"
    if has_ab:
        lines += [
            f"- **Compared:** base `{args.base[:12]}` and {pr}.",
            f"- **Method:** each kernel ran {rounds} times on each side, back to back on one GPU, alternating which "
            "side went first. *Change* is the median over the runs, in throughput: 10% slower means 10% less work "
            f"per second. A kernel is flagged only when it changed by {args.threshold:.0%} or more and every run "
            "agreed on the direction.",
        ]
        noise = noise_line(rows)
        if noise:
            lines.append(f"- {noise}")
    else:
        lines.append(f"- **Compared:** {pr} with the nightly benchmark, because the base build did not run. That "
                     "also includes other changes on main and machine-to-machine noise, so only kernels below 80% of "
                     "the nightly count.")
    side, runner_ratio = context
    if baseline_date and runner_ratio is not None:
        what = "base" if side == "base" else "this PR"
        gap = change_text(runner_ratio)
        gap = "within 1% of" if gap == "no change" else f"{gap} than"
        lines.append(f"- **Nightly reference:** this machine measured {what} {gap} the nightly "
                     f"`{baseline_date}.{args.platform}.json` (median over kernels"
                     f"{nightly_age(args.base_date, baseline_date)}).")
    elif not baseline_date:
        lines.append(f"- **Nightly reference:** none for {args.platform} on the release.")
    failed_runs = [r for r in results.get("runs", []) if r.get("error")]
    if failed_runs:
        lines.append("- **Runs that produced nothing:** " +
                     "; ".join(f"{r['side']} run {r['round']}: {r['error']}" for r in failed_runs) + ".")
    only = {state: sum(r["state"] == state for r in rows) for state in ("only-head", "only-base")}
    if has_ab and (only["only-head"] or only["only-base"]):
        lines.append(f"- **Kernel list changed:** {only['only-head']} new on this PR, {only['only-base']} gone.")
    if args.run_url:
        lines.append(f"- [Workflow run]({args.run_url})")
    return lines


def render(args, results, rows, groups, verdict, has_ab, baseline_date, context):
    icon, title = headline(rows, verdict, has_ab)
    lines = [f"### {icon} {display_name(args.platform)}: {title}", ""]
    merged = bool(args.pr_head) and args.pr_head != args.head
    if not results.get("runs"):
        status = f" (bench job: {args.bench_status})" if args.bench_status else ""
        run = f" See the [workflow run]({args.run_url})." if args.run_url else ""
        lines.append(f"The benchmark produced no results{status}, so there is nothing to compare.{run}")
        return "\n".join(lines) + "\n"
    if has_ab:
        lines += findings(rows, args.threshold)
        if groups:
            lines += ["", "| Kernel family | Count | Result |", "|---|---:|---|"]
            for g in groups:
                members = [r for r in rows if r["group"] == g["group"] and r["state"] != "only-base"]
                lines.append(f"| `{op_name(g['group'])}` | {len(members)} | {group_result(members)} |")
    else:
        low = sum(r["nightly_ratio"] is not None and r["nightly_ratio"] < 0.80 for r in rows)
        line = f"{plural(low, 'kernel')} below 80% of the nightly." if low else "No kernel is below 80% of the nightly."
        if low and verdict == "PASS":
            line += " It takes two such kernels, or one below 60%, to flag this looser comparison."
        lines.append(line)
    lines += [""] + kernel_table(rows, has_ab)
    lines += ["", "<details><summary>How this was measured</summary>", ""]
    lines += method_lines(args, results, rows, has_ab, baseline_date, context, merged)
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def report(args):
    try:
        results = json.loads(Path(args.results).read_text())
    except (OSError, ValueError):
        results = {"runs": [], "env": {}}
    rows = compare(results, args.threshold)
    has_ab = any(r["ratio"] is not None for r in rows)
    baseline_date, baseline_path = pick_baseline(args.history, args.base_date.replace("-", ""), args.platform)
    nightly = load_nightly(baseline_path) if baseline_path else {}
    context = add_nightly(rows, nightly, has_ab)
    groups = summarize_groups(rows)
    verdict = verdict_for(rows, has_ab)
    markdown = render(args, results, rows, groups, verdict, has_ab, baseline_date, context)
    Path(args.out_md).write_text(markdown)
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "platform": args.platform, "verdict": verdict, "same_runner": has_ab, "base": args.base, "head":
                    args.head, "threshold": args.threshold, "nightly": baseline_date, "groups": groups, "cases": rows
                }, indent=1))
    print(f"perf smoke [{args.platform}]: {verdict}")
    return 0


def perf_irrelevant(path):
    """Whether a changed path cannot move kernel performance: tests, CI or docs."""
    parts = path.split("/")
    return bool({"test", "tests"} & set(parts[:-1])) or parts[0] in (".github", "docs") or path.endswith(".md")


def relevant(args):
    """Print true when a path changed from BASE to HEAD can move performance, else false."""
    out = subprocess.run(["git", "diff", "--name-only", args.base, args.head], cwd=args.repo, check=True,
                         capture_output=True, text=True).stdout
    paths = [line for line in out.splitlines() if line]
    print("true" if not paths or any(not perf_irrelevant(path) for path in paths) else "false")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("relevant", help="whether BASE..HEAD changes anything that can move performance")
    v.add_argument("--repo", default=".")
    v.add_argument("--base", required=True)
    v.add_argument("--head", required=True)
    b = sub.add_parser("bench", help="run the A/B on this GPU runner")
    b.add_argument("--platform", required=True)
    b.add_argument("--repo", default=".", help="checkout whose python/test/tlx_benchmark harness measures both wheels")
    b.add_argument("--base-wheel", default="", help="blank: head only, compared with the nightly in the report")
    b.add_argument("--head-wheel", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--rounds", type=int, default=3)
    b.add_argument("--select", default=SMOKE_SELECT, help="pytest -k expression over the bench modules")
    b.add_argument("--head-cases", type=int, default=SMOKE_HEAD, help="first N cases per op and direction")
    b.add_argument("--gpu", default="auto", help="physical GPU index; 'auto' picks one and keeps it for every run")
    b.add_argument("--case-timeout", type=int, default=1200, help="pytest-timeout per bench module, seconds")
    b.add_argument("--run-timeout", type=int, default=1800, help="whole pytest run, seconds")
    r = sub.add_parser("report", help="compare and write the markdown section")
    r.add_argument("--results", required=True, help="results.json from bench")
    r.add_argument("--platform", required=True)
    r.add_argument("--history", default="", help="directory of nightly YYYYMMDD.<platform>.json summaries")
    r.add_argument("--base-date", required=True, help="the base commit's date, YYYYMMDD")
    r.add_argument("--base", required=True)
    r.add_argument("--head", required=True)
    r.add_argument("--pr-head", default="")
    r.add_argument("--run-url", default="")
    r.add_argument("--bench-status", default="", help="the bench job's result, quoted when it produced nothing")
    r.add_argument("--threshold", type=float, default=0.05)
    r.add_argument("--out-md", required=True)
    r.add_argument("--out-json", default="")
    args = parser.parse_args(argv)
    return {"bench": bench, "report": report, "relevant": relevant}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
