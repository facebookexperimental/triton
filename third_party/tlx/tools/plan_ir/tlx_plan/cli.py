"""Command line interface for baseline capture, extraction, and replay."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from .baseline import FA_BWD_D128_CASES, FA_BWD_D128_SCHEDULES, make_manifest, read_manifest, write_catalog
from .audit import audit_markdown, audit_plan
from .model import PlanBundle, canonical_json
from .replay import compare_plans, replay_normalized, verify_replay
from .ttgir import extract_plan, normalize_ttgir


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tlx-plan")
    commands = parser.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser("catalog", help="write the fixed M1.1 FA-backward catalog")
    catalog.add_argument("--output", type=Path, required=True)

    manifest = commands.add_parser("manifest", help="write a baseline manifest")
    manifest.add_argument("--case", choices=sorted(FA_BWD_D128_CASES), required=True)
    manifest.add_argument("--schedule", choices=sorted(FA_BWD_D128_SCHEDULES), required=True)
    manifest.add_argument("--source-root", type=Path, required=True)
    manifest.add_argument("--compiler-root", type=Path)
    manifest.add_argument("--output", type=Path, required=True)

    extract = commands.add_parser("extract", help="extract PlanBundle JSON from final TTGIR")
    extract.add_argument("--ttgir", type=Path, required=True)
    extract.add_argument("--manifest", type=Path)
    extract.add_argument(
        "--value-graph",
        type=Path,
        help="native plan-value-graph JSON dumped from final structured TTGIR",
    )
    extract.add_argument("--output", type=Path, required=True)

    normalize = commands.add_parser("normalize", help="remove locations and alpha-rename TTGIR")
    normalize.add_argument("--ttgir", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify", help="verify TTGIR against a PlanBundle")
    verify.add_argument("--ttgir", type=Path, required=True)
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--value-graph", type=Path)
    verify.add_argument("--output", type=Path)

    replay = commands.add_parser("replay", help="normalize TTGIR and prove exact-plan replay")
    replay.add_argument("--ttgir", type=Path, required=True)
    replay.add_argument("--plan", type=Path, required=True)
    replay.add_argument("--value-graph", type=Path)
    replay.add_argument("--normalized-output", type=Path, required=True)
    replay.add_argument("--report", type=Path, required=True)

    diff = commands.add_parser("diff", help="compare two PlanBundles by semantic layer")
    diff.add_argument("expected", type=Path)
    diff.add_argument("actual", type=Path)
    diff.add_argument("--output", type=Path)

    audit = commands.add_parser("audit", help="run the strict M1.4d PlanBundle audit")
    audit.add_argument("--plan", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--markdown-output", type=Path)
    return parser


def _emit(value: object, output: Path | None) -> None:
    payload = canonical_json(value)
    if output:
        output.write_text(payload)
    else:
        print(payload, end="")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "catalog":
        write_catalog(args.output)
    elif args.command == "manifest":
        compiler_root = args.compiler_root or args.source_root
        manifest = make_manifest(
            args.case,
            args.schedule,
            _revision(args.source_root),
            _revision(compiler_root),
        )
        args.output.write_text(canonical_json(manifest.to_dict()))
    elif args.command == "extract":
        manifest = read_manifest(args.manifest) if args.manifest else None
        value_graph = json.loads(args.value_graph.read_text()) if args.value_graph else None
        plan = extract_plan(
            args.ttgir.read_text(),
            manifest=manifest,
            source_name=str(args.ttgir),
            native_value_graph=value_graph,
        )
        plan.write(args.output)
    elif args.command == "normalize":
        args.output.write_text(normalize_ttgir(args.ttgir.read_text()))
    elif args.command == "verify":
        value_graph = json.loads(args.value_graph.read_text()) if args.value_graph else None
        report = verify_replay(
            args.ttgir.read_text(),
            PlanBundle.read(args.plan),
            native_value_graph=value_graph,
        )
        _emit(report, args.output)
        return 0 if report["semantic_match"] else 1
    elif args.command == "replay":
        value_graph = json.loads(args.value_graph.read_text()) if args.value_graph else None
        normalized, report = replay_normalized(
            args.ttgir.read_text(),
            PlanBundle.read(args.plan),
            native_value_graph=value_graph,
        )
        args.normalized_output.write_text(normalized)
        args.report.write_text(canonical_json(report))
        return 0 if report["semantic_match"] else 1
    elif args.command == "diff":
        _emit(compare_plans(PlanBundle.read(args.expected), PlanBundle.read(args.actual)), args.output)
    elif args.command == "audit":
        report = audit_plan(PlanBundle.read(args.plan))
        _emit(report, args.output)
        if args.markdown_output:
            args.markdown_output.write_text(audit_markdown(report))
        return 0 if report["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
