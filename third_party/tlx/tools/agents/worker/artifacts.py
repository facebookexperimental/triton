from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..manager.models import JsonValue, PriorExperimentEvidence, PriorRunEvidence, to_json_value
from .source import source_digest, source_diff

_INLINE_PROFILE_LIMIT_BYTES = 1_000_000
_MAX_PRIOR_JSON_BYTES = 8 * 1024 * 1024
_MAX_PRIOR_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_PRIOR_EXPERIMENTS = 1000
_MAX_PRIOR_PROMPT_EXPERIMENTS = 12
_SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def _bounded_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\x00", " ").split())[:limit]


def load_prior_run_evidence(path: Path) -> PriorRunEvidence:
    resolved = path.resolve()
    experiments_path = resolved / "experiments.json" if resolved.is_dir() else resolved
    run_path = experiments_path.parent
    if not experiments_path.is_file():
        raise ValueError(f"experiments.json not found at {experiments_path}")
    if experiments_path.stat().st_size > _MAX_PRIOR_JSON_BYTES:
        raise ValueError("experiments.json exceeds the prior-run size limit")
    try:
        payload = json.loads(experiments_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read experiments.json: {error}") from error
    if not isinstance(payload, list):
        raise ValueError("experiments.json must contain a list")
    if len(payload) > _MAX_PRIOR_EXPERIMENTS:
        raise ValueError("experiments.json exceeds the experiment count limit")

    hashes: list[str] = []
    evidence: list[PriorExperimentEvidence] = []
    warnings: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"experiment {index} must be an object")
        experiment_id = item.get("experiment_id")
        if not isinstance(experiment_id, str) or not _SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
            raise ValueError(f"experiment {index} has an unsafe experiment_id")
        source_path = (run_path / "experiments" / experiment_id / "kernel.py").resolve()
        try:
            source_path.relative_to(run_path)
        except ValueError as error:
            raise ValueError(f"experiment {experiment_id} escapes the prior run") from error
        if source_path.is_file():
            if source_path.stat().st_size > _MAX_PRIOR_SOURCE_BYTES:
                raise ValueError(f"source for {experiment_id} exceeds the size limit")
            try:
                source = source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise ValueError(f"unable to read source for {experiment_id}: {error}") from error
            if source.strip():
                hashes.append(source_digest(source))
        elif experiment_id != "baseline":
            warnings.append(f"source unavailable for {experiment_id}")

        if experiment_id == "baseline" or len(evidence) >= _MAX_PRIOR_PROMPT_EXPERIMENTS:
            continue
        performance = item.get("performance")
        speedup = performance.get("aggregate_speedup") if isinstance(performance, dict) else None
        if not isinstance(speedup, (int, float)) or not math.isfinite(float(speedup)):
            speedup = None
        evidence.append(
            PriorExperimentEvidence(
                experiment_id=experiment_id,
                status=_bounded_text(item.get("status"), 40) or "unknown",
                hypothesis=_bounded_text(item.get("hypothesis"), 240),
                change=_bounded_text(item.get("mutation_summary"), 240),
                aggregate_speedup=float(speedup) if speedup is not None else None,
                diagnostics=_bounded_text(item.get("diagnostics"), 500),
            )
        )
    return PriorRunEvidence(
        run_path=run_path,
        experiments_path=experiments_path,
        source_hashes=tuple(dict.fromkeys(hashes)),
        experiments=tuple(evidence),
        warnings=tuple(warnings[:20]),
    )


@dataclass(frozen=True)
class CandidateArtifactPaths:
    source_path: Path
    incremental_patch_path: Path
    cumulative_patch_path: Path


@dataclass(frozen=True)
class ArtifactStore:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def write_source(self, experiment_id: str, source: str) -> Path:
        experiment_dir = self.root / "experiments" / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        source_path = experiment_dir / "kernel.py"
        source_path.write_text(source)
        return source_path

    def write_candidate_artifacts(
        self,
        experiment_id: str,
        *,
        source: str,
        parent_source: str,
        parent_id: str,
        baseline_source: str,
    ) -> CandidateArtifactPaths:
        experiment_dir = self.root / "experiments" / experiment_id
        source_path = self.write_source(experiment_id, source)
        incremental_patch_path = experiment_dir / "incremental.patch"
        incremental_patch_path.write_text(
            source_diff(
                parent_source,
                source,
                fromfile=f"experiments/{parent_id}/kernel.py",
                tofile=f"experiments/{experiment_id}/kernel.py",
            )
        )
        cumulative_patch_path = experiment_dir / "cumulative.patch"
        cumulative_patch_path.write_text(
            source_diff(
                baseline_source,
                source,
                fromfile="experiments/baseline/kernel.py",
                tofile=f"experiments/{experiment_id}/kernel.py",
            )
        )
        return CandidateArtifactPaths(
            source_path=source_path,
            incremental_patch_path=incremental_patch_path,
            cumulative_patch_path=cumulative_patch_path,
        )

    def write_json(self, relative_path: str, value: JsonValue | object) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json_value(value), indent=2, sort_keys=True) + "\n")
        return path

    def write_profile(self, experiment_id: str, profile: Mapping[str, JsonValue]) -> Path:
        """Write a per-experiment profile, spilling large payloads to artifacts/."""
        encoded = json.dumps(profile, indent=2, sort_keys=True)
        if len(encoded) > _INLINE_PROFILE_LIMIT_BYTES:
            # Spill to artifacts/profile_traces/ and keep a pointer in the experiment dir.
            trace_path = self.root / "artifacts" / "profile_traces" / f"{experiment_id}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(encoded + "\n")
            pointer: Mapping[str, JsonValue] = {
                "artifact": str(trace_path.relative_to(self.root)),
                "size_bytes": len(encoded),
                "truncated": True,
            }
            return self.write_json(f"experiments/{experiment_id}/profile.json", pointer)
        return self.write_json(f"experiments/{experiment_id}/profile.json", profile)

    def write_aggregated_profile(
        self, name: str, profiles: Mapping[str, Mapping[str, JsonValue]]
    ) -> Path:
        """Write an aggregated per-case profile map (e.g. baseline_profile.json)."""
        encoded = json.dumps(profiles, indent=2, sort_keys=True)
        if len(encoded) > _INLINE_PROFILE_LIMIT_BYTES:
            trace_path = self.root / "artifacts" / "profile_traces" / f"{name}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(encoded + "\n")
            pointer: Mapping[str, JsonValue] = {
                "artifact": str(trace_path.relative_to(self.root)),
                "size_bytes": len(encoded),
                "truncated": True,
            }
            return self.write_json(f"{name}.json", pointer)
        return self.write_json(f"{name}.json", profiles)

    def write_best(self, source: str) -> Path:
        path = self.root / "best_kernel.py"
        path.write_text(source)
        return path
