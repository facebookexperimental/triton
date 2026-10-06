"""Validate the generated sub-tiled backward DDG's defining shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_generated(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if "@generated" not in document:
        raise SystemExit(f"{path} has no @generated marker")
    return document


def _only_loop(document: dict[str, Any], path: Path) -> dict[str, Any]:
    loops = document.get("loops", [])
    if len(loops) != 1 or loops[0].get("loop_id") != 0:
        raise SystemExit(f"{path} must contain exactly loop 0")
    return loops[0]


def _node_identity(node: dict[str, Any]) -> tuple[int, str, str]:
    return node["id"], node["op_ref"], node["op_kind"]


def _edge_identity(edge: dict[str, Any]) -> tuple[int, int, str, int, int]:
    # ScheduleGraph records carriedness through distance and calls these data.
    kind = "data" if edge["kind"] == "loop_carried" else edge["kind"]
    return edge["src"], edge["dst"], kind, edge["distance"], edge["latency"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ddg", nargs="?", type=Path, default=Path("ddg.json"))
    parser.add_argument("--schedule", type=Path)
    args = parser.parse_args()
    schedule_path = args.schedule or args.ddg.with_name("schedule_graph.json")
    document = _load_generated(args.ddg)
    schedule = _load_generated(schedule_path)
    if document["kernel"]["name"] != "fa_bwd_dkdv_subtiled":
        raise SystemExit(f"unexpected kernel in {args.ddg}")
    if schedule["kernel"]["name"] != document["kernel"]["name"]:
        raise SystemExit("DDG and schedule graph name different kernels")

    ddg = _only_loop(document, args.ddg)["ddg"]
    schedule_loop = _only_loop(schedule, schedule_path)["schedule_loop"]
    ddg_nodes = ddg["nodes"]
    schedule_nodes = schedule_loop["graph"]["nodes"]
    if [_node_identity(node) for node in ddg_nodes] != [
        _node_identity(node) for node in schedule_nodes
    ]:
        raise SystemExit("schedule graph does not cover the DDG nodes exactly")
    if [_edge_identity(edge) for edge in ddg["edges"]] != [
        _edge_identity(edge) for edge in schedule_loop["graph"]["edges"]
    ]:
        raise SystemExit("schedule graph edges differ from the DDG")
    if not {node["op_ref"] for node in ddg_nodes} <= document["ops"].keys():
        raise SystemExit("DDG references an operation absent from the ops table")

    counts = {
        kind: sum(node["op_kind"] == kind for node in ddg_nodes)
        for kind in ("math.exp2", "ttng.tc_gen5_mma")
    }
    expected = {"math.exp2": 2, "ttng.tc_gen5_mma": 10}
    if counts != expected:
        raise SystemExit(f"unexpected loop structure: {counts}, expected {expected}")
    print(f"verified {args.ddg} and {schedule_path}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
