"""Enumerate and run the AutoWS schedule x memory-space x memory-plan space.

The child command must compile exactly one searched loop and append its
compiler records to ``TRITON_WS_SEARCH_MANIFEST``. A zero exit status is the
candidate's correctness/validation result. If ``--metric-regex`` is provided,
the final capture group in the child's stdout is recorded as its metric.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass
class RunResult:
    schedule_rank: int
    memory_space_rank: int
    memory_rank: int
    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str
    records: list[dict[str, Any]]


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid manifest JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict) or record.get("kind") not in {"schedule", "memory-space", "memory"}:
            raise RuntimeError(f"invalid search record at {path}:{line_number}")
        records.append(record)
    return records


def _contiguous_ranks(records: Sequence[dict[str, Any]], kind: str) -> list[int]:
    ranks = sorted({int(record["rank"]) for record in records if record.get("kind") == kind})
    if ranks and ranks != list(range(ranks[-1] + 1)):
        raise RuntimeError(f"non-contiguous {kind} ranks in manifest: {ranks}")
    return ranks


def _memory_ranks(records: Sequence[dict[str, Any]], schedule_rank: int) -> list[int]:
    matching = [
        record for record in records
        if record.get("kind") == "memory" and int(record.get("schedule_pick", -1)) == schedule_rank
    ]
    ranks = _contiguous_ranks(matching, "memory")
    return ranks or [0]


def _memory_space_ranks(records: Sequence[dict[str, Any]]) -> list[int]:
    ranks = _contiguous_ranks(records, "memory-space")
    return ranks or [0]


def _selected(records: Sequence[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("kind") == kind and record.get("selected") is True]


def _run_candidate(command: Sequence[str], base_env: dict[str, str], manifest: Path, schedule_rank: int,
                   memory_space_rank: int, memory_rank: int, schedule_topk: int, memory_space_topk: int,
                   memory_topk: int, timeout: float | None) -> RunResult:
    manifest.unlink(missing_ok=True)
    env = base_env.copy()
    env.update({
        "TRITON_ALWAYS_COMPILE": "1",
        "TRITON_USE_META_WS": "1",
        "TRITON_USE_MODULO_SCHEDULE": "contracted",
        "TRITON_MODULO_TOPK": str(schedule_topk),
        "TRITON_MODULO_PICK": str(schedule_rank),
        "TRITON_WS_MEMORY_SPACE_TOPK": str(memory_space_topk),
        "TRITON_WS_MEMORY_SPACE_PICK": str(memory_space_rank),
        "TRITON_WS_SMEM_PLAN_SEARCH": "1",
        "TRITON_WS_MEM_PLAN_TOPK": str(memory_topk),
        "TRITON_WS_MEM_PLAN_PICK": str(memory_rank),
        "TRITON_WS_SEARCH_MANIFEST": str(manifest),
    })
    start = time.perf_counter()
    completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout, check=False)
    elapsed = time.perf_counter() - start
    return RunResult(schedule_rank, memory_space_rank, memory_rank, completed.returncode, elapsed, completed.stdout,
                     completed.stderr, _read_manifest(manifest))


def _parse_env(values: Sequence[str]) -> dict[str, str]:
    result = os.environ.copy()
    for value in values:
        key, separator, setting = value.partition("=")
        if not separator or not key:
            raise ValueError(f"expected KEY=VALUE for --set-env, got {value!r}")
        result[key] = setting
    return result


def _metric(stdout: str, pattern: re.Pattern[str] | None) -> float | None:
    if pattern is None:
        return None
    matches = list(pattern.finditer(stdout))
    if not matches:
        return None
    match = matches[-1]
    value = match.group(1) if match.lastindex else match.group(0)
    return float(value)


def _selection_error(run: RunResult) -> str | None:
    selected_schedules = {int(record["rank"]) for record in _selected(run.records, "schedule")}
    if selected_schedules != {run.schedule_rank}:
        return f"selected schedule ranks {sorted(selected_schedules)}; expected [{run.schedule_rank}]"

    memory_space_records = [record for record in run.records if record.get("kind") == "memory-space"]
    if memory_space_records:
        available = _contiguous_ranks(memory_space_records, "memory-space")
        expected = min(run.memory_space_rank, available[-1])
        selected = {int(record["rank"]) for record in memory_space_records if record.get("selected") is True}
        if selected != {expected}:
            return f"selected memory-space ranks {sorted(selected)}; expected [{expected}]"

    memory_records = [record for record in run.records if record.get("kind") == "memory"]
    pools = {str(record["pool"]) for record in memory_records}
    for pool in pools:
        pool_records = [record for record in memory_records if record["pool"] == pool]
        available = _contiguous_ranks(pool_records, "memory")
        expected = min(run.memory_rank, available[-1])
        selected = {int(record["rank"]) for record in pool_records if record.get("selected") is True}
        if selected != {expected}:
            return f"selected {pool} ranks {sorted(selected)}; expected [{expected}]"
    return None


def _result_record(run: RunResult, pattern: re.Pattern[str] | None) -> dict[str, Any]:
    metric = _metric(run.stdout, pattern)
    status = "passed" if run.returncode == 0 else "failed"
    selection_error = None if status == "failed" else _selection_error(run)
    if selection_error is not None:
        status = "manifest-mismatch"
    if status == "passed" and pattern is not None and metric is None:
        status = "metric-missing"
    return {
        "schedule_rank": run.schedule_rank,
        "memory_space_rank": run.memory_space_rank,
        "memory_rank": run.memory_rank,
        "status": status,
        "returncode": run.returncode,
        "elapsed_seconds": round(run.elapsed_seconds, 6),
        "metric": metric,
        "error": selection_error,
        "schedule": _selected(run.records, "schedule"),
        "memory_space": _selected(run.records, "memory-space"),
        "memory": _selected(run.records, "memory"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-topk", type=int, default=4)
    parser.add_argument("--memory-space-topk", type=int, default=1)
    parser.add_argument("--memory-topk", type=int, default=4)
    parser.add_argument("--results", type=Path, required=True, help="JSONL output path")
    parser.add_argument("--metric-regex", help="Regex whose final match and first capture group is a metric")
    parser.add_argument("--timeout", type=float, default=None, help="Per-candidate timeout in seconds")
    parser.add_argument("--set-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to compile, validate, and measure")
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a child command is required after --")
    if args.schedule_topk < 1 or args.memory_space_topk < 1 or args.memory_topk < 1:
        parser.error("top-K values must be positive")

    try:
        base_env = _parse_env(args.set_env)
        pattern = re.compile(args.metric_regex) if args.metric_regex else None
    except (ValueError, re.error) as exc:
        parser.error(str(exc))

    args.results.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with tempfile.TemporaryDirectory(prefix="triton-autows-search-") as temp_dir, args.results.open("w") as output:
        manifest = Path(temp_dir) / "manifest.jsonl"
        first = _run_candidate(command, base_env, manifest, 0, 0, 0, args.schedule_topk, args.memory_space_topk,
                               args.memory_topk, args.timeout)
        schedule_ranks = _contiguous_ranks(first.records, "schedule")
        if not schedule_ranks:
            raise RuntimeError("child command emitted no schedule records")
        memory_space_ranks = _memory_space_ranks(first.records)

        for schedule_rank in schedule_ranks:
            for memory_space_rank in memory_space_ranks:
                discovery = first if schedule_rank == 0 and memory_space_rank == 0 else _run_candidate(
                    command, base_env, manifest, schedule_rank, memory_space_rank, 0, args.schedule_topk,
                    args.memory_space_topk, args.memory_topk, args.timeout)
                memory_ranks = _memory_ranks(discovery.records, schedule_rank)
                for memory_rank in memory_ranks:
                    run = discovery if memory_rank == 0 else _run_candidate(
                        command, base_env, manifest, schedule_rank, memory_space_rank, memory_rank, args.schedule_topk,
                        args.memory_space_topk, args.memory_topk, args.timeout)
                    record = _result_record(run, pattern)
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                    if record["status"] != "passed":
                        failures += 1
                        if run.stderr:
                            print(run.stderr, end="", file=os.sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
