"""Summarize the samples written by usage_monitor.sh as a Markdown report.

Reports GPU utilization, GPU memory and power, CPU cores busy and pod memory
over the whole step, then a per-minute timeline. GPU utilization is
nvidia-smi's utilization.gpu or rocm-smi's "GPU use (%)": the share of time the
GPU was busy, not how many SMs or CUs were busy.
"""

import argparse
import csv
import math

GIB = 1024**3


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def series(rows):
    """Turn raw samples into per-sample values; CPU busy needs two samples."""
    out = []
    prev = None
    for row in rows:
        t = number(row.get("time"))
        if t is None:
            continue
        usage = number(row.get("cpu_usage_usec"))
        cores = None
        if prev is not None and usage is not None and prev[1] is not None and t > prev[0]:
            cores = (usage - prev[1]) / 1e6 / (t - prev[0])
        prev = (t, usage)
        memory = number(row.get("memory_bytes"))
        gpu_memory = number(row.get("gpu_memory_mib"))
        out.append({
            "time": t,
            "gpu_util": number(row.get("gpu_util_pct")),
            "gpu_memory": gpu_memory / 1024 if gpu_memory is not None else None,
            "gpu_power": number(row.get("gpu_power_w")),
            "cpu_cores": cores,
            "memory": memory / GIB if memory is not None else None,
        })
    return out


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return (sum(values) / len(values), percentile(values, 0.5), percentile(values, 0.9), max(values))


METRICS = [
    ("gpu_util", "GPU utilization (%)"),
    ("gpu_memory", "GPU memory used (GiB)"),
    ("gpu_power", "GPU power (W)"),
    ("cpu_cores", "CPU busy (cores)"),
    ("memory", "Pod memory (GiB)"),
]


def cgroup_bytes(value):
    """A cgroup memory file's value for the report; empty means the file is missing."""
    if not value:
        return "n/a (no cgroup file)"
    if value == "max":
        return "none"
    return f"{int(value) / GIB:.1f} GiB"


def report(samples, cpus, gpus, memory_max, memory_peak):
    if len(samples) < 2:
        return "### Resource usage\n\nFewer than two samples; nothing to report.\n"
    seconds = samples[-1]["time"] - samples[0]["time"]
    lines = [
        "### Resource usage",
        "",
        f"{len(samples)} samples over {seconds:.0f} s. Pod: {cpus} CPUs, {gpus} visible GPU(s), "
        f"memory limit {cgroup_bytes(memory_max)}, memory peak {cgroup_bytes(memory_peak)}. GPU columns are GPU 0.",
        "",
        "| Metric | mean | p50 | p90 | max |",
        "|---|---|---|---|---|",
    ]
    for key, label in METRICS:
        s = stats(x[key] for x in samples)
        cells = " | ".join(f"{v:.1f}" for v in s) if s else "n/a | n/a | n/a | n/a"
        lines.append(f"| {label} | {cells} |")
    util = [x["gpu_util"] for x in samples if x["gpu_util"] is not None]
    if util:
        busy = 100 * sum(u >= 90 for u in util) / len(util)
        idle = 100 * sum(u == 0 for u in util) / len(util)
        lines += ["", f"GPU utilization was at least 90% in {busy:.0f}% of samples and 0% in {idle:.0f}%."]
    lines += [
        "",
        "Per minute (means; pod memory is the max):",
        "",
        "| Minute | GPU util (%) | GPU mem (GiB) | CPU busy (cores) | Pod mem (GiB) |",
        "|---|---|---|---|---|",
    ]
    start = samples[0]["time"]
    minutes = {}
    for x in samples:
        minutes.setdefault(int((x["time"] - start) // 60), []).append(x)
    for minute, group in sorted(minutes.items()):
        cells = []
        for key, pick in (("gpu_util", 0), ("gpu_memory", 0), ("cpu_cores", 0), ("memory", 3)):
            s = stats(x[key] for x in group)
            cells.append(f"{s[pick]:.1f}" if s else "n/a")
        lines.append(f"| {minute} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--cpus", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--memory-max", default="")
    parser.add_argument("--memory-peak", default="")
    args = parser.parse_args(argv)
    try:
        samples = series(load(args.csv))
    except OSError as e:
        print(f"### Resource usage\n\nNo samples: {e}\n")
        return 0
    print(report(samples, args.cpus, args.gpus, args.memory_max, args.memory_peak), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
