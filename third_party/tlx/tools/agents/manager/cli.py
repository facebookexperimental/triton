from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any, Callable, TextIO
from xml.sax.saxutils import escape

from ..build_agent.harness import HarnessExecutionError, SubprocessHarness
from ..build_agent.vcs import (
    AutoCommitSession,
    commit_promotion,
    commit_rollback,
    failed_auto_commit,
    prepare_auto_commit,
)
from ..worker.artifacts import load_prior_run_evidence
from ..worker.providers import CodexCandidateProvider, MockLLMProvider
from .models import (
    AutoCommitResult,
    ExperimentSummary,
    InputCase,
    KernelOptimizationRequest,
    KernelTarget,
    OptimizationBudget,
    PerformanceSummary,
    passes_protected_cases,
    to_json_value,
)
from .optimizer import KernelOptimizer


class _TeeTextIO:
    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self._terminal = terminal
        self._log = log

    def write(self, text: str) -> int:
        self._terminal.write(text)
        self._log.write(text)
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._terminal.isatty()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._terminal, name)


def _load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a Triton or TLX kernel with a deterministic harness.")
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument(
        "--reference-kernel", type=Path, default=None, help=
        "Optional reference kernel source used as correctness oracle (harness verify can compare candidate vs reference)."
    )
    parser.add_argument("--harness", type=Path, default=None)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument(
        "--arch",
        default=None,
        help=
        "Target arch under validator/targets/<arch>/<kernel> (e.g. blackwell, hopper, host). Defaults to first available.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prior-run",
        type=Path,
        default=None,
        help=(
            "Read-only path to a prior TLX Agent output directory or its "
            "experiments.json; imports evidence and source hashes without "
            "adopting the prior winner."
        ),
    )
    parser.add_argument(
        "--result-format",
        choices=["full", "summary", "none"],
        default="full",
        help=("Final stdout payload: full JSON (default), a compact JSON summary, or nothing. "
              "Full results are always persisted under --output-dir."),
    )
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=2)
    parser.add_argument("--max-candidate-seconds", type=float, default=600.0)
    parser.add_argument("--max-total-seconds", type=float, default=3600.0)
    parser.add_argument("--min-speedup", type=float, default=1.01)
    parser.add_argument("--max-cv", type=float, default=0.10)
    parser.add_argument("--benchmark-repetitions", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--commit-winner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Commit a successfully revalidated winner to the kernel's repository "
              "(default: enabled; use --no-commit-winner to disable)."),
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit subject; correctness, performance, rationale, and generic attribution are added to the body.",
    )
    parser.add_argument(
        "--vcs",
        choices=["auto", "git", "hg"],
        default="auto",
        help="Version control for --commit-winner; auto detects from --kernel.",
    )
    parser.add_argument(
        "--provider",
        choices=["codex", "mock"],
        default="codex",
        help="Candidate provider: codex (default) or mock (deterministic CI stub).",
    )
    parser.add_argument(
        "--harness-mode",
        choices=["subprocess", "standalone"],
        default="subprocess",
        help="subprocess (default, isolated) or standalone (in-process, for debugging).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect harness profile() for baseline and candidates (default: profile is always collected).",
    )
    parser.add_argument(
        "--diagnostic-proton-intra-kernel",
        action="store_true",
        default=False,
        help=("Collect diagnostic-only per-warp proton_intra_kernel traces for the "
              "baseline and final winner only."),
    )
    parser.add_argument(
        "--budget",
        type=Path,
        default=None,
        help="Optional JSON file that overrides --max-* / --min-speedup / --max-cv flags.",
    )
    return parser.parse_args(arguments)


def _budget_from_args(args: argparse.Namespace) -> OptimizationBudget:
    if args.budget is not None:
        payload = _load_json(args.budget)
        return OptimizationBudget(
            max_rounds=int(payload.get("max_rounds", args.max_rounds)),
            candidates_per_round=int(payload.get("candidates_per_round", args.candidates_per_round)),
            max_candidate_seconds=float(payload.get("max_candidate_seconds", args.max_candidate_seconds)),
            max_total_seconds=float(payload.get("max_total_seconds", args.max_total_seconds)),
            min_speedup=float(payload.get("min_speedup", args.min_speedup)),
            max_cv=float(payload.get("max_cv", args.max_cv)),
            benchmark_repetitions=int(payload.get("benchmark_repetitions", args.benchmark_repetitions)),
        )
    return OptimizationBudget(
        max_rounds=args.max_rounds,
        candidates_per_round=args.candidates_per_round,
        max_candidate_seconds=args.max_candidate_seconds,
        max_total_seconds=args.max_total_seconds,
        min_speedup=args.min_speedup,
        max_cv=args.max_cv,
        benchmark_repetitions=args.benchmark_repetitions,
    )


def _resolve_harness_paths(kernel: Path, harness: Path | None, cases: Path | None, target: Path | None,
                           arch: str | None) -> tuple[Path, Path, Path]:
    # Kernel-only invocation: infer harness/cases/target from
    # validator/targets/<arch>/<stem>/
    # e.g. --kernel gemm.py -> validator/targets/blackwell/gemm/{harness.py,cases.json,target.json}
    # Harness must be colocated with cases (target-specific), so both are resolved together.
    base = Path(__file__).resolve().parents[1] / "validator" / "targets"
    stem = kernel.stem  # gemm, vector_add, etc.
    if base.exists() and (harness is None or cases is None or target is None):
        archs = sorted(p.name for p in base.iterdir() if p.is_dir() and (p / stem).is_dir())
        chosen = arch or (archs[0] if archs else None)
        if chosen is None:
            chosen = "blackwell" if harness is None else None
        if chosen is not None:
            tdir = base / chosen / stem
            if harness is None and (tdir / "harness.py").exists():
                harness = tdir / "harness.py"
            if cases is None and (tdir / "cases.json").exists():
                cases = tdir / "cases.json"
            if target is None and (tdir / "target.json").exists():
                target = tdir / "target.json"
    if harness is None or cases is None or target is None:
        missing = [n for n, v in [("harness", harness), ("cases", cases), ("target", target)] if v is None]
        raise SystemExit(
            f"missing required {'/'.join(missing)}; pass them explicitly or use a kernel with validator/targets/<arch>/<name>/"
        )
    return harness, cases, target


# TL strategy and curated knowledge live in the frozen bundle, not a global doc
# tree, so a run's prompt is reproducible from its recorded bundle hashes. Per
# `references/input-contract.md` the provider receives only their resolved text.
# Backstop against a runaway document, not a budget to fill -- the prompt also
# carries the preamble, reference kernel and profiles. ~2x the shipped gfx942
# bundle. Truncation is announced so a fragment is never read as complete.
MAX_GUIDANCE_BYTES = 16384


def _resolve_guidance(target_path: Path, inline: str) -> str:
    """Bundle knowledge files plus ``target.json``'s inline string, widest first."""
    target_dir = target_path.resolve().parent
    arch = target_dir.parent.name
    kernel = target_dir.name
    # validator/targets/<arch>/<kernel>/target.json -> agents/
    agents_dir = target_dir.parents[3]
    sections: list[str] = []
    for path in (
        agents_dir / "tl" / "strategies" / arch / f"{kernel}.md",
        agents_dir / "knowledge_keeper" / "knowledge" / arch / "architecture.md",
        agents_dir / "knowledge_keeper" / "knowledge" / arch / "targets" / f"{kernel}.md",
    ):
        if path.is_file():
            text = path.read_text().strip()
            if text:
                sections.append(f"--- {path.name} ({path.parent.name}) ---\n{text}")
    if inline.strip():
        sections.append(inline.strip())
    guidance = "\n\n".join(sections)
    if len(guidance) > MAX_GUIDANCE_BYTES:
        guidance = (guidance[:MAX_GUIDANCE_BYTES] +
                    f"\n\n[truncated at {MAX_GUIDANCE_BYTES} bytes; read the bundle files for the rest]")
    return guidance


def _expected_cuda_major(arch: str) -> int | None:
    normalized = arch.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"hopper", "h100", "sm90", "sm_90"}:
        return 9
    if normalized in {"blackwell", "b200", "gb200", "sm100", "sm_100"}:
        return 10
    return None


def _expected_gcn_arch(arch: str) -> str | None:
    normalized = arch.lower().replace("-", "_").replace(" ", "_")
    return {
        "gfx942": "gfx942",
        "mi300": "gfx942",
        "mi300x": "gfx942",
        "cdna3": "gfx942",
        "gfx950": "gfx950",
        "mi350": "gfx950",
        "mi355": "gfx950",
        "cdna4": "gfx950",
        "gfx1250": "gfx1250",
    }.get(normalized)


def _probe_gcn_arch(device: str | None) -> str:
    try:
        import torch
    except ImportError as error:
        raise SystemExit("HIP target validation requires torch to be importable") from error
    if not torch.cuda.is_available():
        raise SystemExit("HIP target selected, but no ROCm device is available")
    if not getattr(torch.version, "hip", None):
        raise SystemExit("HIP target selected, but this torch is not a ROCm build")
    torch_device = torch.device(device or "cuda")
    index = torch_device.index
    if index is None:
        index = torch.cuda.current_device()
    # e.g. "gfx942:sramecc+:xnack-" -- the feature suffix is not part of the target.
    return torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]


def _probe_cuda_compute_capability(device: str | None) -> tuple[int, int]:
    try:
        import torch
    except ImportError as error:
        raise SystemExit("CUDA target validation requires torch to be importable") from error
    if not torch.cuda.is_available():
        raise SystemExit("CUDA target selected, but no CUDA device is available")
    torch_device = torch.device(device or "cuda")
    if torch_device.type != "cuda":
        raise SystemExit(f"CUDA target selected, but target device is {device!r}")
    device_index = torch_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return torch.cuda.get_device_capability(device_index)


def _validate_host_matches_target(
    target: KernelTarget,
    arch: str | None,
    capability_probe: Callable[[str | None], tuple[int, int]] = _probe_cuda_compute_capability,
    gcn_probe: Callable[[str | None], str] = _probe_gcn_arch,
) -> None:
    if target.backend not in {"cuda", "hip"}:
        return
    requested = arch or target.architecture
    if target.backend == "cuda":
        expected: object | None = _expected_cuda_major(requested)
    else:
        expected = _expected_gcn_arch(requested)
    if expected is None:
        return
    previous_environment: dict[str, str | None] = {}
    try:
        for key, value in target.environment.items():
            previous_environment[key] = os.environ.get(key)
            os.environ[key] = value
        if target.backend == "cuda":
            actual_major, actual_minor = capability_probe(target.device)
            actual: str = f"sm_{actual_major}{actual_minor}"
            matched = actual_major == expected
            expected_label = f"sm_{expected}x"
        else:
            actual = gcn_probe(target.device)
            matched = actual == expected
            expected_label = str(expected)
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if not matched:
        raise SystemExit(f"--arch {requested} expects {expected_label}, "
                         f"but {target.device or target.backend} is {actual}")


def _performance_commit_body(
    baseline_summary: PerformanceSummary,
    comparison: PerformanceSummary,
    experiment_id: str,
    commit_summary: str,
    heading: str,
) -> str:
    baseline_by_id = {case.case_id: case for case in baseline_summary.cases}
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for winner in comparison.cases:
        baseline = baseline_by_id.get(winner.case_id)
        baseline_timing = baseline.timing if baseline else None
        winner_timing = winner.timing
        if baseline_timing is not None and winner_timing is not None:
            speedup = baseline_timing.median_us / winner_timing.median_us
            baseline_us = f"{baseline_timing.median_us:.2f}"
            winner_us = f"{winner_timing.median_us:.2f}"
            speedup_text = f"{speedup:.4f}x"
            baseline_cv = f"{100.0 * baseline_timing.coefficient_of_variation:.2f}%"
            winner_cv = f"{100.0 * winner_timing.coefficient_of_variation:.2f}%"
        else:
            baseline_us = winner_us = speedup_text = baseline_cv = winner_cv = "n/a"
        rows.append(
            (
                winner.case_id,
                baseline_us,
                winner_us,
                speedup_text,
                baseline_cv,
                winner_cv,
                "pass" if winner.verification.passed else "fail",
            )
        )

    headers = ("Case", "Baseline us", "Winner us", "Speedup", "Base CV", "Winner CV", "Correct")
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()

    table = [format_row(headers), format_row(tuple("-" * width for width in widths))]
    table.extend(format_row(row) for row in rows)
    validation = (
        "Performance:\n"
        f"{heading} for {experiment_id}:\n"
        + "\n".join(table)
        + f"\nWeighted aggregate speedup: {comparison.aggregate_speedup:.4f}x."
    )
    summary = commit_summary.strip()
    return f"{summary}\n\n{validation}" if summary else validation


class _PromotionAutoCommitter:
    def __init__(
        self,
        session: AutoCommitSession,
        harness_path: Path,
        cases: tuple[InputCase, ...],
        target: KernelTarget,
        budget: OptimizationBudget,
        output_dir: Path,
        fallback_subject: str,
        override_subject: str | None,
    ) -> None:
        self._session = session
        self._harness_path = harness_path
        self._cases = cases
        self._target = target
        self._budget = budget
        self._output_dir = output_dir
        self._fallback_subject = fallback_subject
        self._override_subject = override_subject

    def _validate(self, committed_source: str, experiment_id: str) -> None:
        validation = SubprocessHarness(
            self._harness_path, self._budget.max_candidate_seconds
        ).evaluate(
            committed_source,
            self._cases,
            self._target,
            self._budget.benchmark_repetitions,
        )
        self._output_dir.joinpath(
            "experiments", experiment_id, "commit_revalidation.json"
        ).write_text(json.dumps(to_json_value(validation), indent=2, sort_keys=True) + "\n")
        if not passes_protected_cases(validation, self._cases):
            raise HarnessExecutionError(
                "merged promotion source failed one or more protected correctness cases"
            )

    def commit_promotion(
        self,
        experiment: ExperimentSummary,
        source: str,
        baseline: PerformanceSummary,
        performance: PerformanceSummary,
    ) -> AutoCommitResult:
        subject = self._override_subject or experiment.commit_title or self._fallback_subject
        try:
            result = commit_promotion(
                self._session,
                source,
                subject,
                _performance_commit_body(
                    baseline,
                    performance,
                    experiment.experiment_id,
                    experiment.commit_summary,
                    "Promotion evaluation",
                ),
                validate_committed_source=lambda committed: self._validate(
                    committed, experiment.experiment_id
                ),
            )
        except Exception as error:  # noqa: BLE001
            result = failed_auto_commit(self._session.snapshot, subject, error)
        _report_commit(result)
        self._output_dir.joinpath("promotion_commits.json").write_text(
            json.dumps(
                to_json_value(tuple(self._session.promotion_commits)),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return result

    def rollback_to_baseline(self, diagnostics: str) -> AutoCommitResult:
        subject = "Revert TLX agent promotions after failed final revalidation"
        try:
            result = commit_rollback(self._session, subject, diagnostics)
        except Exception as error:  # noqa: BLE001
            result = failed_auto_commit(self._session.snapshot, subject, error)
        _report_commit(result)
        self._output_dir.joinpath("rollback_commit.json").write_text(
            json.dumps(to_json_value(result), indent=2, sort_keys=True) + "\n"
        )
        return result


def _report_commit(commit_result: object) -> None:
    result = commit_result
    parts = [
        "[tlx-agent] commit",
        f"status={'committed' if result.success else 'failed'}",
        f"vcs={result.vcs or 'unknown'}",
    ]
    if result.commit_revision:
        parts.append(f"id={result.commit_revision}")
    if result.repo_root:
        parts.append(f"repo={result.repo_root}")
    if result.target_relpath:
        parts.append(f"file={result.target_relpath}")
    if result.subject:
        parts.append(f"subject={json.dumps(result.subject)}")
    parts.append(f"attribution={json.dumps(result.attribution)}")
    if result.diagnostics:
        parts.append(f"diagnostics={json.dumps(result.diagnostics)}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def _primary_case(cases: tuple[InputCase, ...]) -> InputCase | None:
    if not cases:
        return None
    return max(enumerate(cases), key=lambda item: (item[1].weight, -item[0]))[1]


def _review_title(
    kernel_path: Path,
    target: KernelTarget,
    cases: tuple[InputCase, ...],
) -> str:
    operation = kernel_path.parent.name
    primary = _primary_case(cases)
    parameters = primary.parameters if primary is not None else {}
    dimensions = [parameters.get(name) for name in ("m", "n", "k")]
    if all(isinstance(value, int | float) for value in dimensions):
        shape = "×".join(str(int(value)) for value in dimensions)
        return f"{target.architecture} {operation} {shape}"
    return f"{target.architecture} {operation}"


def _winning_lineage(result: Any) -> list[Any]:
    by_id = {
        experiment.experiment_id: experiment
        for experiment in result.experiments
    }
    winner = None
    for experiment in reversed(result.experiments):
        if experiment.status != "promoted" or not experiment.source_path.is_file():
            continue
        if experiment.source_path.read_text() == result.best_kernel:
            winner = experiment
            break
    lineage: list[Any] = []
    while winner is not None and winner.experiment_id != "baseline":
        lineage.append(winner)
        winner = by_id.get(winner.parent_id)
    lineage.reverse()
    return lineage


def _review_metadata(
    result: Any,
    cases: tuple[InputCase, ...],
    title: str,
    output_dir: Path,
) -> dict[str, Any]:
    baseline_by_id = {case.case_id: case for case in result.baseline.cases}
    final_by_id = {case.case_id: case for case in result.final.cases}
    passed = sum(case.verification.passed for case in result.final.cases)
    total = len(result.final.cases)
    correctness = f"{'PASS' if passed == total else 'FAIL'} ({passed}/{total} cases)"

    performance: list[dict[str, Any]] = []
    for case in cases:
        before = baseline_by_id.get(case.case_id)
        after = final_by_id.get(case.case_id)
        before_us = before.timing.median_us if before and before.timing else None
        after_us = after.timing.median_us if after and after.timing else None
        speedup = before_us / after_us if before_us and after_us else None
        performance.append(
            {
                "case_id": case.case_id,
                "before_us": before_us,
                "after_us": after_us,
                "speedup": speedup,
                "protected": case.protected,
            }
        )

    primary = _primary_case(cases)
    primary_performance = next(
        (
            row
            for row in performance
            if primary is not None and row["case_id"] == primary.case_id
        ),
        None,
    )
    lineage = _winning_lineage(result)
    what_changed = [
        experiment.mutation_summary
        for experiment in lineage
        if experiment.mutation_summary
    ]
    why_it_works = [
        " ".join(
            part
            for part in (
                experiment.hypothesis,
                experiment.expected_effect,
            )
            if part
        )
        for experiment in lineage
    ]
    why_it_works = [item for item in why_it_works if item]
    evidence = [experiment.evidence for experiment in lineage if experiment.evidence]
    return {
        "title": title,
        "correctness_signoff": correctness,
        "primary_case": primary.case_id if primary is not None else None,
        "primary_performance": primary_performance,
        "aggregate_speedup": result.final.aggregate_speedup,
        "what_changed": what_changed,
        "why_it_works": why_it_works,
        "evidence": evidence,
        "per_case_performance": performance,
        "swimlane": str(output_dir / "swimlane.svg"),
        "knowledge": "Include the human-approved Knowledge Keeper patch in the kernel PR.",
    }


def _format_review_summary(review: dict[str, Any]) -> str:
    lines = [str(review["title"]), "", "What changed"]
    lines.extend(f"- {item}" for item in review["what_changed"] or ["No winner was promoted."])
    lines.extend(("", "Why it works"))
    lines.extend(f"- {item}" for item in review["why_it_works"] or ["No confirmed mechanism."])
    if review["evidence"]:
        lines.extend(("", "Evidence"))
        lines.extend(f"- {item}" for item in review["evidence"])
    lines.extend(("", "Correctness", f"- {review['correctness_signoff']}", "", "Performance"))
    primary = review["primary_performance"]
    if primary and primary["before_us"] is not None and primary["after_us"] is not None:
        lines.append(
            f"- {primary['case_id']}: {primary['before_us']:.3f} us -> "
            f"{primary['after_us']:.3f} us ({primary['speedup']:.4f}x)"
        )
    lines.append(f"- Weighted aggregate: {review['aggregate_speedup']:.4f}x")
    lines.extend(
        f"- {row['case_id']}: {row['before_us']:.3f} us -> {row['after_us']:.3f} us "
        f"({row['speedup']:.4f}x)"
        for row in review["per_case_performance"]
        if row["before_us"] is not None and row["after_us"] is not None
    )
    lines.extend(
        (
            "",
            "Review artifacts",
            f"- Swimlane: {review['swimlane']}",
            f"- Knowledge: {review['knowledge']}",
        )
    )
    return "\n".join(lines) + "\n"


def _commit_body(review: dict[str, Any]) -> str:
    primary = review["primary_performance"]
    performance = "Performance: unavailable"
    if primary and primary["before_us"] is not None and primary["after_us"] is not None:
        performance = (
            f"Performance: {primary['case_id']} {primary['before_us']:.3f} us -> "
            f"{primary['after_us']:.3f} us ({primary['speedup']:.4f}x)"
        )
    what = " ".join(review["what_changed"]) or "No promoted source change."
    why = " ".join(review["why_it_works"]) or "No confirmed mechanism."
    return "\n".join(
        (
            f"What changed: {what}",
            f"Why it works: {why}",
            f"Correctness: {review['correctness_signoff']}",
            performance,
            f"Weighted aggregate: {review['aggregate_speedup']:.4f}x",
        )
    )


def _svg_lines(
    x: int,
    y: int,
    text: object,
    *,
    css_class: str = "card-text",
    width: int = 30,
    limit: int = 2,
) -> str:
    value = " ".join(str(text).split())
    lines = textwrap.wrap(value, width=width, break_long_words=False) or [""]
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = textwrap.shorten(
            lines[-1], width=width, placeholder="…"
        )
    return "".join(
        f'<text x="{x}" y="{y + index * 18}" class="{css_class}">'
        f"{escape(line)}</text>"
        for index, line in enumerate(lines)
    )


def _write_swimlane(
    review: dict[str, Any], output_dir: Path, result: Any | None = None
) -> Path:
    path = output_dir / "swimlane.svg"
    primary = review["primary_performance"]
    performance = "No promoted performance result"
    if primary and primary["before_us"] is not None and primary["after_us"] is not None:
        performance = (
            f"{primary['before_us']:.3f} us → {primary['after_us']:.3f} us · "
            f"{primary['speedup']:.4f}x"
        )
    what = " ".join(review["what_changed"]) or "No promoted source change"
    why = " ".join(review["why_it_works"]) or "No confirmed mechanism"
    experiments = (
        [item for item in result.experiments if item.experiment_id != "baseline"]
        if result is not None
        else []
    )
    baseline_us = primary.get("before_us") if primary else None
    primary_case_id = review.get("primary_case")
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        evaluations = experiment.performance.cases if experiment.performance else ()
        evaluation = next(
            (
                item
                for item in evaluations
                if item.case_id == primary_case_id
            ),
            evaluations[0] if evaluations else None,
        )
        passed = sum(item.verification.passed for item in evaluations)
        total = len(evaluations)
        latency_us = (
            evaluation.timing.median_us
            if evaluation is not None and evaluation.timing is not None
            else None
        )
        speedup = baseline_us / latency_us if baseline_us and latency_us else None
        rows.append(
            {
                "id": experiment.experiment_id,
                "status": experiment.status.upper(),
                "hypothesis": experiment.hypothesis or "No hypothesis recorded",
                "change": experiment.mutation_summary or "No source change",
                "scope": experiment.change_scope or "unknown scope",
                "passed": passed,
                "total": total,
                "latency_us": latency_us,
                "speedup": speedup,
            }
        )

    profile_lines = ["Profiler evidence persisted"]
    if result is not None:
        baseline_evaluation = next(
            (
                item
                for item in result.baseline.cases
                if item.case_id == primary_case_id
            ),
            result.baseline.cases[0] if result.baseline.cases else None,
        )
        att = (
            baseline_evaluation.profile.get("att", {})
            if baseline_evaluation is not None
            else {}
        )
        if isinstance(att, dict) and att.get("mode") == "att":
            profile_lines = [f"ATT decoded · {att.get('instruction_rows', 0)} rows"]
            top_stalls = att.get("stall_by_opcode", [])
            if isinstance(top_stalls, list):
                profile_lines.extend(
                    f"{item.get('opcode', 'unknown')} {item.get('stall_pct', 0):.1f}%"
                    for item in top_stalls[:2]
                    if isinstance(item, dict)
                )

    lane_specs = (
        (30, "MANAGER · JUDGE", "scope · budget · verdict", "#ef6f91", "#2c1420"),
        (260, "BUILD · WITCH", "environment · VCS", "#f3a846", "#2d2110"),
        (490, "KNOWLEDGE · PRIEST", "human-approved guidance", "#e979b4", "#301827"),
        (720, "PROFILER · SEER", "raw evidence only", "#aa7cff", "#211934"),
        (950, "TL · SHERIFF", "reason · combine · PR plan", "#55a4ff", "#10233b"),
        (1180, "WORKER · VILLAGER", "isolated implementation", "#38c9c4", "#102e30"),
        (1410, "CORRECTNESS", "numerical gate", "#40c58a", "#122b20"),
        (1640, "PERFORMANCE", "all supplied shapes", "#a7cf4b", "#263011"),
    )
    row_start = 330
    row_pitch = 138
    final_y = row_start + len(rows) * row_pitch + 28
    height = max(1040, final_y + 190)
    lane_height = height - 170

    lanes = []
    headers = []
    for x, name, detail, color, fill in lane_specs:
        lanes.append(
            f'<rect x="{x}" y="105" width="220" height="{lane_height}" '
            f'rx="10" fill="{fill}" stroke="{color}" class="lane"/>'
        )
        headers.append(
            f'<rect x="{x}" y="105" width="220" height="70" rx="10" fill="{color}"/>'
            f'<text x="{x + 15}" y="134" class="header">{escape(name)}</text>'
            f'<text x="{x + 15}" y="155" class="header-sub">{escape(detail)}</text>'
        )

    row_svg = []
    for index, row in enumerate(rows):
        y = row_start + index * row_pitch
        correct = row["total"] > 0 and row["passed"] == row["total"]
        status_class = "pass" if row["status"] == "PROMOTED" else "reject"
        correctness_class = "pass" if correct else "reject"
        perf_text = (
            f"{row['latency_us']:.3f} us · {row['speedup']:.4f}x"
            if row["latency_us"] is not None and row["speedup"] is not None
            else "blocked · no timing"
        )
        row_svg.append(
            f'<rect x="962" y="{y}" width="196" height="104" rx="8" fill="#1d4e7d" stroke="#55a4ff" class="card"/>'
            f'<text x="974" y="{y + 23}" class="card-title">{escape(row["id"])} · TL FINDING</text>'
            f'{_svg_lines(974, y + 45, row["hypothesis"])}'
            f'<text x="974" y="{y + 91}" class="{status_class}">{escape(row["status"])}</text>'
            f'<rect x="1192" y="{y}" width="196" height="104" rx="8" fill="#18575a" stroke="#38c9c4" class="card"/>'
            f'<text x="1204" y="{y + 23}" class="card-title">{escape(row["scope"])}</text>'
            f'{_svg_lines(1204, y + 45, row["change"], limit=3)}'
            f'<rect x="1422" y="{y}" width="196" height="104" rx="8" fill="#194d35" stroke="#40c58a" class="card"/>'
            f'<text x="1434" y="{y + 28}" class="card-title">CALLBACK · {"PASS" if correct else "FAIL"}</text>'
            f'<text x="1434" y="{y + 60}" class="metric">{row["passed"]} / {row["total"]}</text>'
            f'<text x="1434" y="{y + 86}" class="{correctness_class}">{"VALIDATED" if correct else "BLOCKED"}</text>'
            f'<rect x="1652" y="{y}" width="196" height="104" rx="8" fill="#4b5b1f" stroke="#a7cf4b" class="card"/>'
            f'<text x="1664" y="{y + 28}" class="card-title">ALL-SHAPE SWEEP</text>'
            f'<text x="1664" y="{y + 58}" class="metric">{escape(perf_text)}</text>'
            f'<text x="1664" y="{y + 86}" class="{status_class}">{escape(row["status"])}</text>'
            f'<path d="M1158 {y + 52}H1182" class="flow"/>'
            f'<path d="M1388 {y + 52}H1412" class="flow"/>'
            f'<path d="M1618 {y + 52}H1642" class="flow"/>'
            f'<path d="M1642 {y + 91}C1440 {y + 121} 1260 {y + 121} 1158 {y + 91}" class="feedback"/>'
        )

    case_count = len(result.baseline.cases) if result is not None else len(review["per_case_performance"])
    stopping_reason = result.stopping_reason if result is not None else "completed"
    winner = result.winner_experiment_id if result is not None else "unknown"
    baseline_text = (
        f"{baseline_us:.3f} us · 1.0000x"
        if baseline_us is not None
        else "timing unavailable"
    )
    final_status = "BASELINE RETAINED" if winner == "baseline" else f"WINNER · {winner}"
    profile_text = " · ".join(profile_lines)

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="{height}" viewBox="0 0 1920 {height}" role="img" aria-labelledby="title description">
  <title id="title">{escape(str(review["title"]))}</title>
  <desc id="description">Chronological swimlane of Manager, Build, Knowledge, Profiler, TL, Worker, Correctness, and Performance actions.</desc>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#a9b7d0"/></marker>
    <marker id="feedback-arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#67a7ff"/></marker>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#000" flood-opacity="0.35"/></filter>
    <style>
      .page-title {{ fill:#f5f8ff; font:800 32px Inter,sans-serif; letter-spacing:1px; }}
      .subtitle {{ fill:#a9b7d0; font:15px Inter,sans-serif; }}
      .header {{ fill:#07101f; font:800 15px Inter,sans-serif; letter-spacing:.6px; }}
      .header-sub {{ fill:#17243a; font:11px Inter,sans-serif; }}
      .card-title {{ fill:#f7f9ff; font:800 13px Inter,sans-serif; }}
      .card-text {{ fill:#d7dfef; font:11px Inter,sans-serif; }}
      .metric {{ fill:#fff; font:800 12px ui-monospace,SFMono-Regular,Menlo,monospace; }}
      .lane {{ stroke-width:1; }}
      .card {{ stroke-width:1.5; filter:url(#shadow); }}
      .flow {{ fill:none; stroke:#a9b7d0; stroke-width:2; marker-end:url(#arrow); }}
      .feedback {{ fill:none; stroke:#67a7ff; stroke-width:2; stroke-dasharray:6 5; marker-end:url(#feedback-arrow); }}
      .divider {{ stroke:#263650; stroke-width:1; stroke-dasharray:4 7; }}
      .pass {{ fill:#42d392; font:800 11px Inter,sans-serif; }}
      .reject {{ fill:#ff7f89; font:800 11px Inter,sans-serif; }}
      .neutral {{ fill:#ffca68; font:800 11px Inter,sans-serif; }}
    </style>
  </defs>
  <rect width="1920" height="{height}" fill="#070b14"/>
  <text x="32" y="47" class="page-title">{escape(str(review["title"]))}</text>
  <text x="32" y="76" class="subtitle">time flows downward · {case_count} supplied shape(s) · {len(rows)} Worker run(s)</text>
  {''.join(lanes)}
  {''.join(headers)}
  <rect x="42" y="195" width="196" height="94" rx="8" fill="#5b263b" stroke="#ef6f91" class="card"/>
  <text x="54" y="219" class="card-title">FREEZE CAMPAIGN</text>
  {_svg_lines(54, 241, f'{case_count} supplied shape(s)')}
  {_svg_lines(54, 265, f'{len(rows)} Worker run(s)')}
  <rect x="272" y="195" width="196" height="94" rx="8" fill="#5a3b12" stroke="#f3a846" class="card"/>
  <text x="284" y="219" class="card-title">PREPARE ENVIRONMENT</text>
  {_svg_lines(284, 241, 'isolated build and VCS state')}
  <rect x="732" y="195" width="196" height="94" rx="8" fill="#4a3372" stroke="#aa7cff" class="card"/>
  <text x="744" y="219" class="card-title">DEEP BASELINE PROFILE</text>
  {_svg_lines(744, 241, profile_text, limit=3)}
  <rect x="1652" y="195" width="196" height="94" rx="8" fill="#4b5b1f" stroke="#a7cf4b" class="card"/>
  <text x="1664" y="219" class="card-title">BASELINE · {case_count}/{case_count}</text>
  <text x="1664" y="250" class="metric">{escape(baseline_text)}</text>
  <path d="M238 242H262" class="flow"/><path d="M468 242H722" class="flow"/><path d="M928 242C1240 242 1380 242 1642 242" class="flow"/>
  <line x1="30" y1="309" x2="1860" y2="309" class="divider"/>
  {''.join(row_svg)}
  <line x1="30" y1="{final_y - 18}" x2="1860" y2="{final_y - 18}" class="divider"/>
  <rect x="42" y="{final_y}" width="196" height="120" rx="8" fill="#5b263b" stroke="#ef6f91" class="card"/>
  <text x="54" y="{final_y + 24}" class="card-title">FINAL VERDICT</text>
  {_svg_lines(54, final_y + 48, final_status, limit=2)}
  {_svg_lines(54, final_y + 88, stopping_reason, limit=2)}
  <rect x="502" y="{final_y}" width="196" height="120" rx="8" fill="#642d52" stroke="#e979b4" class="card"/>
  <text x="514" y="{final_y + 24}" class="card-title">KNOWLEDGE</text>
  {_svg_lines(514, final_y + 48, 'Human approval required before durable guidance changes', limit=4)}
  <rect x="962" y="{final_y}" width="196" height="120" rx="8" fill="#1d4e7d" stroke="#55a4ff" class="card"/>
  <text x="974" y="{final_y + 24}" class="card-title">WHY IT WORKS</text>
  {_svg_lines(974, final_y + 48, why, limit=4)}
  <rect x="1192" y="{final_y}" width="196" height="120" rx="8" fill="#18575a" stroke="#38c9c4" class="card"/>
  <text x="1204" y="{final_y + 24}" class="card-title">WHAT CHANGED</text>
  {_svg_lines(1204, final_y + 48, what, limit=4)}
  <rect x="1422" y="{final_y}" width="196" height="120" rx="8" fill="#194d35" stroke="#40c58a" class="card"/>
  <text x="1434" y="{final_y + 24}" class="card-title">FINAL CORRECTNESS</text>
  {_svg_lines(1434, final_y + 52, review['correctness_signoff'], css_class='metric')}
  <rect x="1652" y="{final_y}" width="196" height="120" rx="8" fill="#4b5b1f" stroke="#a7cf4b" class="card"/>
  <text x="1664" y="{final_y + 24}" class="card-title">FINAL PERFORMANCE</text>
  {_svg_lines(1664, final_y + 52, performance, css_class='metric', limit=3)}
</svg>
'''
    path.write_text(svg)
    return path


def _print_result(
    result: Any,
    output_dir: Path,
    result_format: str,
    review: dict[str, Any] | None = None,
) -> None:
    if result_format == "none":
        return
    if result_format == "full":
        print(json.dumps(to_json_value(result), indent=2, sort_keys=True))
        return
    summary = {
        "artifacts_dir": str(output_dir),
        "manager_log": str(output_dir / "manager.log"),
        "plan_pool": str(output_dir / "tl" / "plan_pool.json"),
        "result_json": str(output_dir / "result.json"),
        "swimlane": str(output_dir / "swimlane.svg"),
        "stopping_reason": result.stopping_reason,
        "success": result.success,
        "final_speedup": result.final.aggregate_speedup,
        "experiments": len(result.experiments),
    }
    if review is not None:
        summary.update(
            {
                "title": review["title"],
                "correctness_signoff": review["correctness_signoff"],
                "primary_performance": review["primary_performance"],
                "what_changed": review["what_changed"],
                "why_it_works": review["why_it_works"],
                "review_summary": str(output_dir / "review_summary.txt"),
            }
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _run(args: argparse.Namespace) -> int:
    harness_path, cases_path, target_path = _resolve_harness_paths(args.kernel, args.harness, args.cases, args.target,
                                                                   args.arch)
    case_payloads = _load_json(cases_path)
    target_payload = _load_json(target_path)
    cases = tuple(
        InputCase(
            case_id=str(case["case_id"]),
            parameters=case.get("parameters", {}),
            weight=float(case.get("weight", 1.0)),
            protected=bool(case.get("protected", True)),
        ) for case in case_payloads)
    target = KernelTarget(
        backend=str(target_payload["backend"]),
        architecture=str(target_payload["architecture"]),
        device=target_payload.get("device"),
        environment=target_payload.get("environment", {}),
        optimization_guidance=_resolve_guidance(target_path, str(target_payload.get("optimization_guidance", ""))),
    )
    _validate_host_matches_target(target, args.arch)
    budget = _budget_from_args(args)
    # CLI always evaluates via the optimizer's SubprocessHarness. The legacy
    # --harness-mode flag is kept for compatibility and documented as such;
    # standalone evaluation is available programmatically via StandaloneHarness.
    if args.harness_mode != "subprocess":
        import sys as _sys

        print(
            "warning: --harness-mode standalone is only available via the Python API "
            "(StandaloneHarness); CLI still uses subprocess isolation.",
            file=_sys.stderr,
        )
    provider = (MockLLMProvider() if args.provider == "mock" else CodexCandidateProvider(
        model=args.model, timeout_seconds=budget.max_candidate_seconds))
    kernel_path = args.kernel.resolve()
    kernel_source = kernel_path.read_text()
    review_title = _review_title(kernel_path, target, cases)
    commit_subject = args.commit_message or review_title
    commit_snapshot = None
    if args.commit_winner:
        try:
            commit_snapshot = prepare_auto_commit(kernel_path, kernel_source, args.vcs)
        except Exception as error:  # noqa: BLE001
            commit_result = failed_auto_commit(None, commit_subject, error)
            _report_commit(commit_result)
            print(json.dumps(to_json_value(commit_result), indent=2, sort_keys=True))
            return 3
    reference_source = args.reference_kernel.read_text() if args.reference_kernel else None
    prior_run_evidence = None
    if args.prior_run is not None:
        try:
            prior_run_evidence = load_prior_run_evidence(args.prior_run)
        except ValueError as error:
            raise SystemExit(f"--prior-run is invalid: {error}") from error
        print(
            "[tlx-agent] prior-run "
            f"path={json.dumps(str(prior_run_evidence.run_path))} "
            f"experiments={len(prior_run_evidence.experiments)} "
            f"source_hashes={len(prior_run_evidence.source_hashes)} "
            f"warnings={len(prior_run_evidence.warnings)}",
            file=sys.stderr,
            flush=True,
        )
        for warning in prior_run_evidence.warnings:
            print(
                f"[tlx-agent] prior-run warning={json.dumps(warning)}",
                file=sys.stderr,
                flush=True,
            )
    request = KernelOptimizationRequest(
        kernel_source=kernel_source,
        reference_kernel_source=reference_source,
        harness_path=harness_path,
        cases=cases,
        target=target,
        budget=budget,
        output_dir=args.output_dir,
        diagnostic_proton_intra_kernel=args.diagnostic_proton_intra_kernel,
        prior_run_evidence=prior_run_evidence,
    )
    promotion_committer = None
    if commit_snapshot is not None:
        promotion_committer = _PromotionAutoCommitter(
            AutoCommitSession.create(commit_snapshot),
            harness_path,
            cases,
            target,
            budget,
            args.output_dir,
            review_title,
            args.commit_message,
        )
    result = KernelOptimizer(provider).optimize(request, promotion_committer)
    review = _review_metadata(result, cases, review_title, args.output_dir)
    swimlane_path = _write_swimlane(review, args.output_dir, result)
    print(
        "[tlx-agent] FINAL_ARTIFACT "
        f"kind=swimlane path={str(swimlane_path.resolve())!r} recipient=human",
        file=sys.stderr,
        flush=True,
    )
    args.output_dir.joinpath("review_summary.txt").write_text(
        _format_review_summary(review)
    )
    exit_code = 0 if result.success else 2
    if result.auto_commit is not None:
        args.output_dir.joinpath("auto_commit.json").write_text(
            json.dumps(to_json_value(result.auto_commit), indent=2, sort_keys=True) + "\n"
        )
    if result.stopping_reason in {"promotion_commit_failed", "rollback_commit_failed"}:
        exit_code = 3
    _print_result(result, args.output_dir, args.result_format, review)
    return exit_code


def main(arguments: list[str] | None = None) -> int:
    args = _parse_args(arguments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "manager.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = _TeeTextIO(stdout, log)
        sys.stderr = _TeeTextIO(stderr, log)
        try:
            print(
                f"[tlx-agent] LOG path={str(log_path.resolve())!r}",
                file=sys.stderr,
                flush=True,
            )
            return _run(args)
        except SystemExit as error:
            if error.code not in (None, 0):
                print(error.code, file=sys.stderr, flush=True)
            return error.code if isinstance(error.code, int) else 1
        except KeyboardInterrupt:
            traceback.print_exc(file=sys.stderr)
            return 130
        except Exception:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            return 1
        finally:
            sys.stdout = stdout
            sys.stderr = stderr


if __name__ == "__main__":
    raise SystemExit(main())
