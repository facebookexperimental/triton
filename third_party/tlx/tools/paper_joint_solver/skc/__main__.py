"""CLI for the Twill-to-manual-CUDA handoff and deterministic audits."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ._schema import SKCError
from .compiler import (
    audit_authoring,
    audit_bundle,
    audit_mapping,
    audit_memory,
    audit_sync,
    prepare_manual_cuda_handoff,
    scaffold_manual_cuda,
)

_COMMANDS = {
    "handoff",
    "scaffold",
    "audit-authoring",
    "audit-bundle",
    "audit-mapping",
    "audit-sync",
    "audit-memory",
}


def _add_handoff_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--solution", required=True)
    parser.add_argument("--ddg", required=True)
    parser.add_argument("--baseline-graph", required=True)
    parser.add_argument("--ir-out", required=True)
    parser.add_argument("--manifest-out", required=True)


def _subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skc")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_handoff_arguments(commands.add_parser("handoff"))
    scaffold = commands.add_parser("scaffold")
    scaffold.add_argument("--ir", "--pipelined-ir", dest="ir", required=True)
    scaffold.add_argument("--handoff", required=True)
    scaffold.add_argument("--out-dir", "--output-dir", dest="out_dir", required=True)
    for name in ("mapping", "sync", "memory"):
        command = commands.add_parser(f"audit-{name}")
        command.add_argument("--ir", "--pipelined-ir", dest="ir", required=True)
        command.add_argument(f"--{name}", f"--{name}-plan", dest=name, required=True)
    authoring = commands.add_parser("audit-authoring")
    authoring.add_argument("--ir", "--pipelined-ir", dest="ir", required=True)
    authoring.add_argument("--handoff", required=True)
    authoring.add_argument("--authoring", required=True)
    bundle = commands.add_parser("audit-bundle")
    bundle.add_argument("--ir", "--pipelined-ir", dest="ir", required=True)
    bundle.add_argument("--handoff", required=True)
    bundle.add_argument("--authoring", required=True)
    bundle.add_argument("--mapping", required=True)
    bundle.add_argument("--memory", "--memory-plan", dest="memory", required=True)
    bundle.add_argument("--sync", "--sync-plan", dest="sync", required=True)
    return parser


def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skc")
    _add_handoff_arguments(parser)
    return parser


def _run_handoff(args: argparse.Namespace) -> None:
    result = prepare_manual_cuda_handoff(
        solution_path=args.solution,
        ddg_path=args.ddg,
        baseline_graph_path=args.baseline_graph,
        ir_out=args.ir_out,
        manifest_out=args.manifest_out,
    )
    sys.stdout.write(
        f"wrote {result.ir_path}\n"
        f"wrote {result.manifest_path}\n"
        f"IR sha256 {result.ir_sha256}\n"
    )


def _run_command(args: argparse.Namespace) -> None:
    if args.command == "handoff":
        _run_handoff(args)
    elif args.command == "scaffold":
        result = scaffold_manual_cuda(args.ir, args.handoff, args.out_dir)
        for path in (
            result.cuda_path,
            result.authoring_path,
            result.mapping_path,
            result.memory_plan_path,
            result.sync_plan_path,
        ):
            sys.stdout.write(f"wrote {path}\n")
    elif args.command == "audit-authoring":
        result = audit_authoring(args.ir, args.handoff, args.authoring)
        sys.stdout.write(f"{result.kind} audit passed ({result.checked} checked)\n")
    elif args.command == "audit-bundle":
        result = audit_bundle(
            args.ir,
            args.handoff,
            args.authoring,
            args.mapping,
            args.memory,
            args.sync,
        )
        sys.stdout.write(f"{result.kind} audit passed ({result.checked} checked)\n")
    else:
        audit = {
            "audit-mapping": audit_mapping,
            "audit-sync": audit_sync,
            "audit-memory": audit_memory,
        }[args.command]
        plan = getattr(args, args.command.removeprefix("audit-").replace("-", "_"))
        result = audit(args.ir, plan)
        sys.stdout.write(f"{result.kind} audit passed ({result.checked} checked)\n")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    is_subcommand = bool(
        not arguments or arguments[0] in _COMMANDS or arguments[0] in {"-h", "--help"}
    )
    parser = _subcommand_parser() if is_subcommand else _legacy_parser()
    args = parser.parse_args(arguments)
    if not is_subcommand:
        args.command = "handoff"
    try:
        _run_command(args)
    except SKCError as error:
        sys.stderr.write(f"skc: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
