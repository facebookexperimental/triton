#!/usr/bin/env python3
"""Collect, validate, package, and inspect rocprofv3 ATT bundles."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
from typing import Any, Iterable, Sequence

DECODER_NAMES = (
    "librocprof-trace-decoder.so*",
    "libatt_decoder_trace.so",
)


class AttError(RuntimeError):
    pass


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _profiler_candidates(override: str | None = None) -> list[Path]:
    raw: list[Path] = []
    if override:
        raw.append(Path(override))
    for variable in ("AMD_ROCPROFV3", "TLX_ROCPROFV3"):
        if os.environ.get(variable):
            raw.append(Path(os.environ[variable]))
    raw.extend(Path(path) for path in glob.glob("/usr/local/fbcode/platform*/lib/rocm-dev/bin/rocprofv3"))
    bundled = Path(sys.base_prefix) / "lib" / "rocm-dev" / "bin" / "rocprofv3"
    raw.append(bundled)
    discovered = shutil.which("rocprofv3")
    if discovered:
        raw.append(Path(discovered))
    raw.extend((Path("/opt/rocm/bin/rocprofv3"), Path("/usr/bin/rocprofv3")))
    return _unique_paths(raw)


def _help_text(profiler: Path) -> str:
    try:
        completed = subprocess.run(
            [str(profiler), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout + completed.stderr


def find_profiler(override: str | None = None) -> tuple[Path, str]:
    failures: list[str] = []
    for candidate in _profiler_candidates(override):
        if not os.access(candidate, os.X_OK):
            failures.append(f"{candidate}: not executable")
            continue
        help_text = _help_text(candidate)
        if "--att" in help_text or "--advanced-thread-trace" in help_text:
            return candidate, help_text
        failures.append(f"{candidate}: ATT options unavailable")
    raise AttError("no ATT-capable rocprofv3 found; " + "; ".join(failures))


def _decoder_candidates(profiler: Path, override: str | None = None) -> list[Path]:
    raw: list[Path] = []
    if override:
        raw.append(Path(override))
    explicit = os.environ.get("ROCPROF_ATT_LIBRARY_PATH")
    if explicit:
        raw.extend(Path(entry) for entry in explicit.split(":") if entry)
    if profiler.parent.name == "bin":
        raw.append(profiler.parent.parent / "lib")
    raw.extend(Path(path) for path in glob.glob("/usr/local/fbcode/platform*/lib/rocm-dev/lib"))
    raw.extend((Path("/opt/rocm/lib"), Path("/opt/rocm/lib/rocprofiler")))
    return _unique_paths(raw)


def _decoder_files(directory: Path) -> list[Path]:
    return sorted({path.resolve() for pattern in DECODER_NAMES for path in directory.glob(pattern) if path.is_file()})


def find_decoder(profiler: Path, override: str | None = None) -> tuple[Path, list[Path]]:
    searched = _decoder_candidates(profiler, override)
    for directory in searched:
        files = _decoder_files(directory)
        if files:
            return directory, files
    raise AttError("no ATT decoder found in " + ", ".join(str(path) for path in searched))


def probe(profiler_override: str | None, decoder_override: str | None) -> dict[str, Any]:
    profiler, help_text = find_profiler(profiler_override)
    decoder_dir, decoder_files = find_decoder(profiler, decoder_override)
    return {
        "profiler": str(profiler),
        "decoder_directory": str(decoder_dir),
        "decoder_files": [str(path) for path in decoder_files],
        "supports_rocm_root": "--rocm-root" in help_text,
        "supports_att": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _nonempty(paths: Iterable[Path]) -> list[Path]:
    return sorted(path for path in paths if path.is_file() and path.stat().st_size > 0)


def validate_bundle(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise AttError(f"trace root is not a directory: {root}")

    errors: list[str] = []
    databases = _nonempty(root.rglob("*_results.db"))
    raw_traces = _nonempty(root.rglob("*.att"))
    code_objects = _nonempty(root.rglob("*.out"))
    stats_files = _nonempty(root.rglob("stats_ui_output_agent_*_dispatch_*.csv"))
    ui_dirs = sorted(path for path in root.rglob("ui_output_agent_*_dispatch_*") if path.is_dir())

    for label, paths in (
        ("results database", databases),
        ("raw ATT trace", raw_traces),
        ("code object", code_objects),
        ("decoded stats CSV", stats_files),
        ("decoded UI directory", ui_dirs),
    ):
        if not paths:
            errors.append(f"missing non-empty {label}")

    for database in databases:
        try:
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            result = connection.execute("PRAGMA integrity_check").fetchone()
            connection.close()
            if result != ("ok", ):
                errors.append(f"SQLite integrity check failed for {database}: {result}")
        except sqlite3.Error as error:
            errors.append(f"could not read {database}: {error}")

    stats_rows: dict[str, int] = {}
    for stats_file in stats_files:
        try:
            with stats_file.open(newline="") as stream:
                rows = sum(1 for _ in csv.reader(stream)) - 1
            stats_rows[str(stats_file.relative_to(root))] = rows
            if rows < 1:
                errors.append(f"stats CSV has no data rows: {stats_file}")
        except (OSError, csv.Error) as error:
            errors.append(f"could not read {stats_file}: {error}")

    ui_summaries: list[dict[str, Any]] = []
    for ui_dir in ui_dirs:
        required = ("code.json", "filenames.json", "occupancy.json")
        for filename in required:
            required_path = ui_dir / filename
            if not required_path.is_file() or required_path.stat().st_size == 0:
                errors.append(f"missing non-empty {filename} in {ui_dir}")
        wstates = _nonempty(ui_dir.glob("wstates*.json"))
        waves = _nonempty(ui_dir.glob("se*_sm*_sl*_wv*.json"))
        if not wstates:
            errors.append(f"missing wstates JSON in {ui_dir}")
        if not waves:
            errors.append(f"missing per-wave JSON in {ui_dir}")
        json_files = _nonempty(ui_dir.glob("*.json"))
        for json_file in json_files:
            try:
                json.loads(json_file.read_text())
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"invalid JSON {json_file}: {error}")
        ui_summaries.append({
            "path": str(ui_dir.relative_to(root)),
            "json_files": len(json_files),
            "wave_files": len(waves),
            "has_snapshots": (ui_dir / "snapshots.json").is_file(),
            "source_files": len(list(ui_dir.glob("source_*"))),
        })

    summary = {
        "valid": not errors,
        "root": str(root),
        "results_databases": [str(path.relative_to(root)) for path in databases],
        "raw_traces": [str(path.relative_to(root)) for path in raw_traces],
        "code_objects": len(code_objects),
        "stats_rows": stats_rows,
        "ui_directories": ui_summaries,
        "errors": errors,
    }
    if errors:
        raise AttError(json.dumps(summary, indent=2))
    return summary


def _collect_command(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    capability = probe(args.profiler, args.decoder_dir)
    profiler = Path(capability["profiler"])
    command = [
        str(profiler),
        "-d",
        str(args.output.resolve()),
        "-o",
        args.name,
        "--att",
        "--att-library-path",
        capability["decoder_directory"],
        "--kernel-trace",
    ]
    if args.activity > 0:
        command.extend(("--att-activity", str(args.activity)))
    command.extend((
        "--att-target-cu",
        str(args.target_cu),
        "--att-shader-engine-mask",
        args.shader_engine_mask,
        "--att-simd-select",
        args.simd_select,
        "--kernel-include-regex",
        args.kernel_regex,
        "--kernel-iteration-range",
        f"[{args.dispatch}]",
    ))
    if args.serialize_all:
        command.append("--att-serialize-all")
    command.extend(("--", *args.application))
    return command, capability


def collect(args: argparse.Namespace) -> dict[str, Any]:
    if not args.application:
        raise AttError("application command is required after --")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise AttError(f"refusing to overwrite non-empty output directory: {output}")

    command, capability = _collect_command(args)
    environment = os.environ.copy()
    if args.gpu is not None:
        # Visibility filters are applied in layers by ROCm. Setting HIP and
        # ROCR to the same physical index can first reduce the machine to one
        # agent and then incorrectly ask HIP for that original index again.
        environment.pop("HIP_VISIBLE_DEVICES", None)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment["ROCR_VISIBLE_DEVICES"] = args.gpu
    environment["ROCPROF_ATT_LIBRARY_PATH"] = capability["decoder_directory"]

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "working_directory": str(args.cwd.resolve()),
        "application": args.application,
        "command": command,
        "command_display": shlex.join(command),
        "gpu": args.gpu,
        "kernel_regex": args.kernel_regex,
        "dispatch": args.dispatch,
        "target_cu": args.target_cu,
        "shader_engine_mask": args.shader_engine_mask,
        "simd_select": args.simd_select,
        "activity": args.activity,
        "capability": capability,
    }
    if args.dry_run:
        manifest["dry_run"] = True
        return manifest

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "att_manifest.json"
    _write_json(manifest_path, manifest)
    completed = subprocess.run(
        command,
        cwd=args.cwd,
        env=environment,
        check=False,
    )
    manifest["returncode"] = completed.returncode
    if completed.returncode != 0:
        _write_json(manifest_path, manifest)
        raise AttError(f"rocprofv3 exited with status {completed.returncode}")

    validation = validate_bundle(output)
    manifest["validation"] = validation
    _write_json(manifest_path, manifest)
    return manifest


def package_bundle(root: Path, archive: Path | None) -> dict[str, Any]:
    root = root.resolve()
    validation = validate_bundle(root)
    destination = (archive.expanduser().resolve() if archive is not None else root.with_name(root.name + ".tar.gz"))
    if destination == root or root in destination.parents:
        raise AttError(f"archive must be outside trace root: {destination}")
    if destination.exists():
        raise AttError(f"refusing to overwrite archive: {destination}")
    with tarfile.open(destination, "w:gz") as output:
        output.add(root, arcname=root.name)
    return {
        "archive": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "validation": validation,
    }


def _select_ui(root: Path, requested: str | None) -> Path:
    if requested:
        candidate = root / requested
        if candidate.is_dir():
            return candidate
        raise AttError(f"UI directory does not exist: {candidate}")
    candidates = sorted(path for path in root.rglob("ui_output_agent_*_dispatch_*") if path.is_dir())
    if len(candidates) != 1:
        raise AttError(f"expected one UI directory, found {len(candidates)}; select one with --ui")
    return candidates[0]


def cycle_window(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    ui_dir = _select_ui(root, args.ui)
    code = json.loads((ui_dir / "code.json").read_text())["code"]

    matches: list[Sequence[Any]]
    if args.code_id is not None:
        matches = [row for row in code if int(row[2]) == args.code_id]
    elif args.loop_pc is not None:
        pc = int(args.loop_pc, 0)
        matches = [row for row in code if int(row[5]) == pc]
    else:
        pattern = re.compile(args.instruction_regex)
        matches = [row for row in code if pattern.search(str(row[0]))]
    if len(matches) != 1:
        rendered = [{"code_id": row[2], "pc": hex(int(row[5])), "instruction": row[0]} for row in matches[:20]]
        raise AttError(f"loop marker matched {len(matches)} instructions: {rendered}")

    marker = matches[0]
    code_id = int(marker[2])
    wave_path = ui_dir / args.wave
    if not wave_path.is_file():
        raise AttError(f"wave JSON does not exist: {wave_path}")
    wave = json.loads(wave_path.read_text())["wave"]
    occurrences = [int(event[0]) for event in wave["instructions"] if int(event[-1]) == code_id]
    occurrences = occurrences[args.occurrence_offset::args.occurrence_stride]

    adjustment = 1 if args.one_based else 0
    first = args.first - adjustment
    last = args.last - adjustment
    if first < 0 or last < first:
        raise AttError("invalid iteration interval")
    if last + 1 >= len(occurrences):
        raise AttError(f"need marker occurrence {last + 1}, but only {len(occurrences)} exist")
    start_cycle = occurrences[first]
    end_cycle = occurrences[last + 1]
    return {
        "root": str(root),
        "ui_directory": str(ui_dir.relative_to(root)),
        "wave": args.wave,
        "numbering": "one-based" if args.one_based else "zero-based",
        "first_iteration": args.first,
        "last_iteration": args.last,
        "start_cycle": start_cycle,
        "end_cycle": end_cycle,
        "duration_cycles": end_cycle - start_cycle,
        "marker": {
            "code_id": code_id,
            "pc": hex(int(marker[5])),
            "instruction": marker[0],
            "raw_occurrences": len([event for event in wave["instructions"] if int(event[-1]) == code_id]),
            "occurrence_stride": args.occurrence_stride,
            "occurrence_offset": args.occurrence_offset,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    probe_parser = subparsers.add_parser("probe", help="find an ATT-capable profiler and decoder")
    probe_parser.add_argument("--profiler")
    probe_parser.add_argument("--decoder-dir")

    collect_parser = subparsers.add_parser("collect", help="collect and validate one ATT dispatch")
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--name", required=True)
    collect_parser.add_argument("--kernel-regex", required=True)
    collect_parser.add_argument("--dispatch", type=int, required=True)
    collect_parser.add_argument("--gpu")
    collect_parser.add_argument("--target-cu", type=int, default=0)
    collect_parser.add_argument("--shader-engine-mask", default="0x1")
    collect_parser.add_argument("--simd-select", default="0xF")
    collect_parser.add_argument("--activity", type=int, default=8)
    collect_parser.add_argument("--serialize-all", action="store_true")
    collect_parser.add_argument("--profiler")
    collect_parser.add_argument("--decoder-dir")
    collect_parser.add_argument("--cwd", type=Path, default=Path.cwd())
    collect_parser.add_argument("--dry-run", action="store_true")
    collect_parser.add_argument("application", nargs=argparse.REMAINDER)

    validate_parser = subparsers.add_parser("validate", help="validate a viewer-ready ATT bundle")
    validate_parser.add_argument("root", type=Path)

    package_parser = subparsers.add_parser("package", help="validate and archive a complete ATT root")
    package_parser.add_argument("root", type=Path)
    package_parser.add_argument("--archive", type=Path)

    window_parser = subparsers.add_parser("window", help="derive a loop-iteration cycle window")
    window_parser.add_argument("root", type=Path)
    window_parser.add_argument("--ui")
    marker = window_parser.add_mutually_exclusive_group(required=True)
    marker.add_argument("--code-id", type=int)
    marker.add_argument("--loop-pc")
    marker.add_argument("--instruction-regex")
    window_parser.add_argument("--wave", default="se0_sm0_sl0_wv0.json")
    window_parser.add_argument("--first", type=int, required=True)
    window_parser.add_argument("--last", type=int, required=True)
    window_parser.add_argument("--one-based", action="store_true")
    window_parser.add_argument("--occurrence-stride", type=int, default=1)
    window_parser.add_argument("--occurrence-offset", type=int, default=0)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.subcommand == "probe":
            result = probe(args.profiler, args.decoder_dir)
        elif args.subcommand == "collect":
            if args.application and args.application[0] == "--":
                args.application = args.application[1:]
            result = collect(args)
        elif args.subcommand == "validate":
            result = validate_bundle(args.root)
        elif args.subcommand == "package":
            result = package_bundle(args.root, args.archive)
        elif args.subcommand == "window":
            result = cycle_window(args)
        else:
            parser.error(f"unknown subcommand: {args.subcommand}")
            return 2
    except (AttError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
