from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .models import JsonValue, to_json_value

_INLINE_PROFILE_LIMIT_BYTES = 1_000_000


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
