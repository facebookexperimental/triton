#!/usr/bin/env python3
"""Test LLM-based modulo scheduling against compiler LIT tests.

Usage:
    python test_llm_schedule.py [--test TEST_NAME] [--model MODEL] [--verbose]

Feeds TTGIR from LIT test files to Claude via the CLI, asks it to produce
a modulo.schedule graph, and validates the output against expected values
extracted from FileCheck lines.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = REPO_ROOT / "test" / "TritonGPU"
PROMPT_FILE = Path(__file__).parent / "scheduling_prompt.md"


@dataclass
class ExpectedSchedule:
    """Expected schedule values extracted from CHECK lines."""

    ii: int | None = None
    max_stage: int | None = None
    prologue_latency: int | None = None
    trip_count: int | None = None
    stages: dict[int, list[dict]] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)
    buffers: list[dict] = field(default_factory=list)


def extract_mlir_input(test_path: Path) -> str:
    """Extract the MLIR input (everything after CHECK lines + function body)."""
    lines = test_path.read_text().splitlines()
    mlir_lines = []
    for line in lines:
        if line.startswith("// RUN:") or line.startswith("// REQUIRES:"):
            continue
        if line.startswith("// CHECK"):
            continue
        if line.startswith("//==="):
            continue
        if line.startswith("// ---"):
            continue
        if line.startswith("//") and not line.startswith("///"):
            continue
        mlir_lines.append(line)
    return "\n".join(mlir_lines).strip()


def extract_expected(test_path: Path) -> ExpectedSchedule:
    """Extract expected values from CHECK lines."""
    text = test_path.read_text()
    expected = ExpectedSchedule()

    ii_match = re.search(r"ii = (\d+)", text)
    if ii_match:
        expected.ii = int(ii_match.group(1))

    stage_match = re.search(r"max_stage = (\d+)", text)
    if stage_match:
        expected.max_stage = int(stage_match.group(1))

    prologue_match = re.search(r"prologue_latency = (\d+)", text)
    if prologue_match:
        expected.prologue_latency = int(prologue_match.group(1))

    trip_match = re.search(r"trip_count = (\d+)", text)
    if trip_match:
        expected.trip_count = int(trip_match.group(1))

    for m in re.finditer(
        r"CHECK.*?modulo\.stage @s(\d+)", text
    ):
        stage_id = int(m.group(1))
        expected.stages[stage_id] = []

    for m in re.finditer(
        r"CHECK.*?(tt\.\w+|ttg\.\w+|ttng\.\w+)\s+\{pipe: (\w+), cycle: (\d+), "
        r"cluster: (\d+), latency: (\d+), selfLatency: (\d+)",
        text,
    ):
        op_name, pipe, cycle, cluster, latency, self_lat = m.groups()
        stage = int(cycle) // expected.ii if expected.ii else 0
        entry = {
            "op": op_name,
            "pipe": pipe,
            "cycle": int(cycle),
            "cluster": int(cluster),
            "latency": int(latency),
            "selfLatency": int(self_lat),
        }
        if stage in expected.stages:
            expected.stages[stage].append(entry)

    for m in re.finditer(
        r"CHECK.*?N(\d+) -> N(\d+)\s+lat=(\d+)\s+dist=(\d+)", text
    ):
        expected.edges.append({
            "src": int(m.group(1)),
            "dst": int(m.group(2)),
            "lat": int(m.group(3)),
            "dist": int(m.group(4)),
        })

    for m in re.finditer(
        r"CHECK.*?%buf(\d+) = modulo\.alloc (\w+) \[(\d+) x", text
    ):
        expected.buffers.append({
            "id": int(m.group(1)),
            "kind": m.group(2),
            "count": int(m.group(3)),
        })

    return expected


def parse_llm_schedule(output: str) -> dict:
    """Parse the modulo.schedule block from LLM output."""
    result = {
        "ii": None,
        "max_stage": None,
        "prologue_latency": None,
        "trip_count": None,
        "stages": {},
        "edges": [],
        "buffers": [],
    }

    schedule_match = re.search(
        r"modulo\.schedule @\w+ \{(.+?)\n\s*\}", output, re.DOTALL
    )
    if not schedule_match:
        return result

    body = schedule_match.group(1)

    header = re.search(
        r"ii = (\d+), max_stage = (\d+)"
        r"(?:, prologue_latency = (\d+))?"
        r"(?:, trip_count = (\d+))?",
        body,
    )
    if header:
        result["ii"] = int(header.group(1))
        result["max_stage"] = int(header.group(2))
        if header.group(3):
            result["prologue_latency"] = int(header.group(3))
        if header.group(4):
            result["trip_count"] = int(header.group(4))

    for m in re.finditer(r"modulo\.stage @s(\d+) \{", body):
        result["stages"][int(m.group(1))] = []

    for m in re.finditer(
        r"(tt\.\w+|ttg\.\w+|ttng\.\w+)\s+\{pipe: (\w+), cycle: (\d+), "
        r"cluster: (\d+), latency: (\d+), selfLatency: (\d+)",
        body,
    ):
        op_name, pipe, cycle, cluster, latency, self_lat = m.groups()
        ii = result["ii"] or 1
        stage = int(cycle) // ii
        entry = {
            "op": op_name,
            "pipe": pipe,
            "cycle": int(cycle),
            "cluster": int(cluster),
            "latency": int(latency),
            "selfLatency": int(self_lat),
        }
        if stage in result["stages"]:
            result["stages"][stage].append(entry)

    for m in re.finditer(
        r"N(\d+) -> N(\d+)\s+lat=(\d+)\s+dist=(\d+)", body
    ):
        result["edges"].append({
            "src": int(m.group(1)),
            "dst": int(m.group(2)),
            "lat": int(m.group(3)),
            "dist": int(m.group(4)),
        })

    for m in re.finditer(
        r"%buf(\d+) = modulo\.alloc (\w+) \[(\d+) x", body
    ):
        result["buffers"].append({
            "id": int(m.group(1)),
            "kind": m.group(2),
            "count": int(m.group(3)),
        })

    return result


def validate(expected: ExpectedSchedule, actual: dict) -> list[str]:
    """Compare actual schedule against expected. Returns list of failures."""
    failures = []

    if expected.ii is not None and actual["ii"] != expected.ii:
        failures.append(f"II: expected {expected.ii}, got {actual['ii']}")

    if expected.max_stage is not None and actual["max_stage"] != expected.max_stage:
        failures.append(
            f"max_stage: expected {expected.max_stage}, got {actual['max_stage']}"
        )

    if (
        expected.prologue_latency is not None
        and actual["prologue_latency"] != expected.prologue_latency
    ):
        failures.append(
            f"prologue_latency: expected {expected.prologue_latency}, "
            f"got {actual['prologue_latency']}"
        )

    if expected.trip_count is not None and actual["trip_count"] != expected.trip_count:
        failures.append(
            f"trip_count: expected {expected.trip_count}, got {actual['trip_count']}"
        )

    for stage_id, expected_nodes in expected.stages.items():
        actual_nodes = actual["stages"].get(stage_id, [])
        for enode in expected_nodes:
            found = False
            for anode in actual_nodes:
                if (
                    anode["op"] == enode["op"]
                    and anode["pipe"] == enode["pipe"]
                    and anode["cycle"] == enode["cycle"]
                    and anode["cluster"] == enode["cluster"]
                ):
                    found = True
                    if anode["latency"] != enode["latency"]:
                        failures.append(
                            f"Stage {stage_id} {enode['op']}: "
                            f"latency expected {enode['latency']}, "
                            f"got {anode['latency']}"
                        )
                    break
            if not found:
                failures.append(
                    f"Stage {stage_id}: missing node {enode['op']} "
                    f"at cycle {enode['cycle']}"
                )

    expected_edge_set = {
        (e["src"], e["dst"], e["lat"], e["dist"]) for e in expected.edges
    }
    actual_edge_set = {
        (e["src"], e["dst"], e["lat"], e["dist"]) for e in actual["edges"]
    }
    for edge in expected_edge_set - actual_edge_set:
        failures.append(f"Missing edge: N{edge[0]} -> N{edge[1]} lat={edge[2]} dist={edge[3]}")

    return failures


def call_claude(mlir_input: str, system_prompt: str, model: str = "sonnet") -> str:
    """Call Claude CLI with the scheduling prompt and MLIR input."""
    user_prompt = (
        "Given the following TTGIR loop body, produce the modulo.schedule "
        "graph by following the steps in your instructions.\n\n"
        "IMPORTANT: Output ONLY the raw modulo.schedule block. Do NOT wrap "
        "it in markdown code fences. Do NOT include any explanation.\n\n"
        "The TTGIR input:\n\n"
        f"{mlir_input}"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as sp_file:
        sp_file.write(system_prompt)
        sp_path = sp_file.name

    try:
        cmd = [
            "claude",
            "--system-prompt", sp_path,
            "-p", user_prompt,
            "--output-format", "text",
            "--model", model,
            "--allowedTools", "",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.stdout
    finally:
        os.unlink(sp_path)


TESTS = {
    "graph": "modulo-schedule-graph.mlir",
    "buffers": "modulo-schedule-graph-buffers.mlir",
    "budget": "modulo-schedule-graph-budget.mlir",
}


def main():
    parser = argparse.ArgumentParser(description="Test LLM modulo scheduling")
    parser.add_argument(
        "--test",
        choices=list(TESTS.keys()) + ["all"],
        default="graph",
        help="Which test to run (default: graph)",
    )
    parser.add_argument(
        "--model",
        default="sonnet",
        help="Claude model to use (default: sonnet)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    system_prompt = PROMPT_FILE.read_text()

    tests_to_run = list(TESTS.keys()) if args.test == "all" else [args.test]

    for test_name in tests_to_run:
        test_path = TEST_DIR / TESTS[test_name]
        print(f"\n{'='*60}")
        print(f"Test: {test_name} ({test_path.name})")
        print(f"{'='*60}")

        mlir_input = extract_mlir_input(test_path)
        expected = extract_expected(test_path)

        if args.verbose:
            print(f"\nExpected: II={expected.ii}, max_stage={expected.max_stage}, "
                  f"prologue_latency={expected.prologue_latency}")
            print(f"Expected stages: {list(expected.stages.keys())}")
            print(f"Expected edges: {len(expected.edges)}")

        print("\nCalling Claude...")
        output = call_claude(mlir_input, system_prompt, model=args.model)

        if args.verbose:
            print(f"\nLLM output:\n{output}")

        actual = parse_llm_schedule(output)

        if actual["ii"] is None:
            print("FAIL: Could not parse modulo.schedule from LLM output")
            if not args.verbose:
                print(f"Output:\n{output[:500]}")
            continue

        print(f"\nParsed: II={actual['ii']}, max_stage={actual['max_stage']}, "
              f"prologue_latency={actual['prologue_latency']}")

        failures = validate(expected, actual)

        if failures:
            print(f"\nFAIL ({len(failures)} issues):")
            for f in failures:
                print(f"  - {f}")
        else:
            print("\nPASS")


if __name__ == "__main__":
    main()
