"""Measured search-space and dispatch-policy tuning for production TLX ops."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..contracts import (
    ExperimentKind,
    InputCase,
    KernelOptimizationRequest,
    KernelTarget,
    OptimizationBudget,
    PerformanceSummary,
    to_json_value,
)
from ..optimizer.agent import CandidateProvider
from ..policy_source import frozen_source_digest
from .benchmark_environment import canonical_arch
from .harness import SubprocessHarness
from .orchestrator import KernelOptimizer
from .policy import is_promotable
from .vcs import commit_winner, prepare_auto_commit

MIN_REASONABLE_FULL_CONFIGS = 16


def infer_kernel_path(repository: Path, op: str, arch: str) -> Path:
    path = repository / "third_party" / "tlx" / "ops" / "kernels" / op / f"{canonical_arch(arch)}.py"
    if not path.is_file():
        raise SystemExit(f"no production TLX kernel at {path}")
    return path


def production_cases(op: str, suite: str) -> tuple[InputCase, ...]:
    shapes = importlib.import_module(f"triton.tlx.ops.kernels.{op}._shapes")
    entries = shapes.FOCUS.resolved_shapes(suite)
    cases = []
    for entry in entries:
        parameters = dict(entry._asdict())
        case_id = _case_id(parameters)
        cases.append(InputCase(case_id=case_id, parameters=parameters))
    if not cases:
        raise SystemExit(f"production suite {suite!r} is empty")
    return tuple(cases)


def run_tuning(
    *,
    repository: Path,
    op: str,
    arch: str,
    suite: str,
    output_dir: Path,
    provider: CandidateProvider,
    budget: OptimizationBudget,
    search_rounds: int,
    heuristic_rounds: int,
    commit: bool,
    commit_message: str | None,
    vcs: str,
    initial_source: str | None = None,
) -> tuple[int, dict[str, Any]]:
    kernel_path = infer_kernel_path(repository, op, arch)
    original_source = initial_source if initial_source is not None else kernel_path.read_text()
    cases = production_cases(op, suite)
    harness_path = Path(__file__).with_name("tuning_harnesses") / f"{op}.py"
    if not harness_path.is_file():
        raise SystemExit(f"tuning does not yet support tlx.ops.{op}")
    search_symbol, heuristic_symbol = _tuning_symbols(original_source)
    target = _tuning_target(op, arch)
    _validate_device_arch(target, arch)
    output_dir.mkdir(parents=True, exist_ok=True)

    search_target = _phase_target(
        target,
        phase="search_space",
        source=original_source,
        editable_symbols=(search_symbol, ),
        guidance=(f"Phase 1/2: improve only {search_symbol}. Preserve every incumbent, kernel, "
                  "dispatch rule, and heuristic_config. Use the measured per-shape top "
                  "configs to add a compact, hardware-valid candidate set; do not author "
                  f"the decision tree yet. If the incumbent full space has fewer than {MIN_REASONABLE_FULL_CONFIGS} "
                  "configurations, expand it to a reasonable, diverse set before pruning."),
        evaluation_policy={
            "profile_complete_per_measurement": True,
            "minimum_full_config_count": MIN_REASONABLE_FULL_CONFIGS,
        },
    )
    search_budget = replace(
        budget,
        max_rounds=search_rounds,
        min_speedup=max(1.0, budget.min_speedup),
    )
    search_request = KernelOptimizationRequest(
        kernel_source=original_source,
        harness_path=harness_path,
        cases=cases,
        target=search_target,
        budget=search_budget,
        output_dir=output_dir / "search_space",
        kernel_path=kernel_path,
        repository_root=repository,
    )
    search_result = KernelOptimizer(provider).optimize(search_request)
    search_source = search_result.best_kernel
    full_config_count = _maximum_full_config_count(search_result.final)
    if full_config_count < MIN_REASONABLE_FULL_CONFIGS:
        output_dir.joinpath("best_kernel.py").write_text(search_source)
        summary = {
            "success": False,
            "task": "tuning",
            "phase": "search_space",
            "op": op,
            "arch": target.architecture,
            "suite": suite,
            "full_config_count": full_config_count,
            "minimum_full_config_count": MIN_REASONABLE_FULL_CONFIGS,
            "diagnostics": "full search space remains unreasonably small",
        }
        result = {
            **summary,
            "summary": summary,
            "search_space": to_json_value(search_result),
            "heuristic": None,
            "commit": None,
        }
        output_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        output_dir.joinpath("result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 2, result

    heuristic_target = _phase_target(
        target,
        phase="heuristic",
        source=search_source,
        editable_symbols=(heuristic_symbol, ),
        guidance=("Phase 2/2: freeze the kernel and full search space. Author only "
                  f"{heuristic_symbol} as a compact decision tree. Optimize measured regret "
                  "against full-space winners, return exactly one config, use general "
                  "dimension/layout predicates, and put one explanatory sentence directly "
                  "before every if branch. Exact-shape predicates are allowed only for a "
                  "measured algorithm switch."),
        evaluation_policy={
            "kind": "full_space_parity",
            "aggregate_min": 0.98,
            "per_case_min": 0.95,
            "max_heuristic_configs": 1,
            "stable_cv_max": 0.03,
            "profile_complete_per_measurement": True,
        },
    )
    heuristic_budget = replace(budget, max_rounds=heuristic_rounds, min_speedup=1.0)
    heuristic_request = KernelOptimizationRequest(
        kernel_source=search_source,
        harness_path=harness_path,
        cases=cases,
        target=heuristic_target,
        budget=heuristic_budget,
        output_dir=output_dir / "heuristic",
        kernel_path=kernel_path,
        repository_root=repository,
    )
    heuristic_result = KernelOptimizer(provider).optimize(heuristic_request)
    passed = is_promotable(
        heuristic_result.final,
        heuristic_budget,
        cases,
        heuristic_target,
    )
    final_source = heuristic_result.best_kernel
    output_dir.joinpath("best_kernel.py").write_text(final_source)
    summary = _parity_summary(heuristic_result.final, cases)
    summary.update({
        "success": passed,
        "task": "tuning",
        "op": op,
        "arch": arch,
        "suite": suite,
        "kernel": str(kernel_path),
        "winner_experiment_id": heuristic_result.winner_experiment_id,
        "heuristic_speedup": heuristic_result.final.aggregate_speedup,
        "best_kernel": str(output_dir / "best_kernel.py"),
        "result": str(output_dir / "result.json"),
    })
    output_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    commit_result = None
    if commit and passed and final_source != original_source:
        snapshot = prepare_auto_commit(kernel_path, original_source, vcs)

        def validate(source: str) -> None:
            performance = SubprocessHarness(harness_path, heuristic_budget.max_candidate_seconds).evaluate(
                source,
                cases,
                heuristic_target,
                heuristic_budget.benchmark_repetitions,
                profile=False,
            )
            if not is_promotable(
                    performance,
                    heuristic_budget,
                    cases,
                    heuristic_target,
            ):
                raise RuntimeError("merged heuristic policy failed parity validation")

        commit_result = commit_winner(
            snapshot,
            final_source,
            commit_message or f"Tune {arch} {op} search space and heuristic policy",
            (f"TLX agent measured suite {suite} and required at least 98% weighted "
             "full-space parity with no stable case below 95%."),
            validate_committed_source=validate,
        )

    result = {
        "success": passed,
        "task": "tuning",
        "op": op,
        "arch": arch,
        "suite": suite,
        "kernel": str(kernel_path),
        "summary": summary,
        "search_space": to_json_value(search_result),
        "heuristic": to_json_value(heuristic_result),
        "commit": to_json_value(commit_result) if commit_result is not None else None,
    }
    output_dir.joinpath("result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return (0 if passed else 2), result


def _parity_summary(
    performance: PerformanceSummary,
    cases: tuple[InputCase, ...],
) -> dict[str, Any]:
    weights = {case.case_id: case.weight for case in cases}
    stable_parities: list[tuple[float, float]] = []
    unstable_count = 0
    max_config_count = 0
    for evaluation in performance.cases:
        metrics = evaluation.verification.metrics
        max_config_count = max(max_config_count, int(metrics.get("heuristic_config_count", 0)))
        parity = metrics.get("full_space_parity")
        if parity is None or not bool(metrics.get("parity_stable", True)):
            unstable_count += 1
            continue
        stable_parities.append((float(parity), float(weights.get(evaluation.case_id, 1.0))))
    total_weight = sum(weight for _, weight in stable_parities)
    aggregate = (math.exp(sum(weight * math.log(parity)
                              for parity, weight in stable_parities) / total_weight) if total_weight else 0.0)
    return {
        "stable_shape_count": len(stable_parities),
        "unstable_shape_count": unstable_count,
        "aggregate_full_space_parity": aggregate,
        "minimum_stable_shape_parity": min((parity for parity, _ in stable_parities), default=0.0),
        "maximum_heuristic_config_count": max_config_count,
    }


def _maximum_full_config_count(performance: PerformanceSummary) -> int:
    return max(
        (int(evaluation.verification.metrics.get("full_config_count", 0)) for evaluation in performance.cases),
        default=0,
    )


def _phase_target(
    target: KernelTarget,
    *,
    phase: str,
    source: str,
    editable_symbols: tuple[str, ...],
    guidance: str,
    evaluation_policy: dict[str, Any] | None = None,
) -> KernelTarget:
    environment = {
        **target.environment,
        "TLX_AGENT_TUNING_PHASE": phase,
        "TLX_AGENT_HEURISTIC_SYMBOL": editable_symbols[0] if phase == "heuristic" else "",
        "TLX_AGENT_EDITABLE_SYMBOLS": ",".join(editable_symbols),
        "TLX_AGENT_FROZEN_SOURCE_DIGEST": frozen_source_digest(source, editable_symbols),
        "TLX_AGENT_PHASE_BASELINE_SHA256": hashlib.sha256(source.encode()).hexdigest(),
    }
    return replace(
        target,
        environment=environment,
        optimization_guidance=f"{target.optimization_guidance}\n\n{guidance}",
        evaluation_policy=evaluation_policy or {},
    )


def _tuning_target(op: str, arch: str) -> KernelTarget:
    architecture = canonical_arch(arch)
    if architecture.startswith("gfx"):
        backend = "hip"
    elif architecture.startswith("sm"):
        backend = "cuda"
    else:
        raise SystemExit(f"cannot infer GPU backend from --arch {arch!r}")
    return KernelTarget(
        backend=backend,
        architecture=architecture,
        device="cuda:0",
        environment={"TLX_AGENT_OP": op},
        optimization_guidance=(
            f"Tune the production {architecture} tlx.ops.{op} implementation from measured evidence. "
            "Preserve correctness, public entry points, and every incumbent configuration."),
        supported_experiment_kinds=(ExperimentKind.PROMOTABLE, ExperimentKind.HUMAN_REVIEW),
    )


def _tuning_symbols(source: str) -> tuple[str, str]:
    tree = ast.parse(source)
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    search_symbol = next(
        (name for name in ("_configs", "get_cuda_autotune_config", "get_autotune_config") if name in functions),
        None,
    )
    if search_symbol is None:
        raise SystemExit("tuning requires a function that constructs the full search space")
    if "heuristic_config" not in functions:
        raise SystemExit("tuning requires heuristic_config() and space='heuristic' support")
    return search_symbol, "heuristic_config"


def _case_id(parameters: dict[str, Any]) -> str:
    a = "_".join(map(str, parameters["a_strides"]))
    b = "_".join(map(str, parameters["b_strides"]))
    return f"{parameters['dtype']}_{parameters['m']}x{parameters['n']}x{parameters['k']}_{a}x{b}"


def _validate_device_arch(target: KernelTarget, arch: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("tuning target requires a GPU")
    device = torch.device(target.device or "cuda")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if target.backend == "cuda":
        expected = canonical_arch(arch)
        actual = f"sm{torch.cuda.get_device_capability(index)[0]}0"
        if expected != actual:
            raise SystemExit(f"--arch {arch} does not match device architecture {actual}")
        return
    if not getattr(torch.version, "hip", None):
        raise SystemExit("tuning target requires a ROCm GPU")
    actual = str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", ""))
    expected_match = re.search(r"gfx[0-9a-f]+", canonical_arch(arch))
    actual_match = re.search(r"gfx[0-9a-f]+", actual.lower())
    if expected_match is None or actual_match is None or expected_match.group(0) != actual_match.group(0):
        raise SystemExit(f"--arch {arch} does not match device architecture {actual or 'unknown'}")
