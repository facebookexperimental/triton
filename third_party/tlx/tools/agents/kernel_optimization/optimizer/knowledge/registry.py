from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ...contracts import KernelTarget


_KNOWLEDGE_ROOT = Path(__file__).resolve().parent


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class KnowledgeBundle:
    priority: int
    backends: tuple[str, ...]
    architectures: tuple[str, ...]
    documents: tuple[Path, ...]

    def applies_to(self, target: KernelTarget) -> bool:
        backend = _normalize(target.backend)
        architecture = _normalize(target.architecture)
        return (not self.backends or backend in self.backends) and (
            not self.architectures or architecture in self.architectures
        )


@lru_cache(maxsize=1)
def knowledge_bundles() -> tuple[KnowledgeBundle, ...]:
    bundles: list[KnowledgeBundle] = []
    for manifest_path in sorted(_KNOWLEDGE_ROOT.glob("**/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundles.append(
            KnowledgeBundle(
                priority=int(payload.get("priority", 0)),
                backends=tuple(_normalize(value) for value in payload.get("backends", ())),
                architectures=tuple(_normalize(value) for value in payload.get("architectures", ())),
                documents=tuple(manifest_path.parent / name for name in payload["documents"]),
            )
        )
    return tuple(sorted(bundles, key=lambda bundle: (bundle.priority, bundle.documents)))


def knowledge_paths(target: KernelTarget) -> tuple[Path, ...]:
    return tuple(
        document for bundle in knowledge_bundles() if bundle.applies_to(target) for document in bundle.documents
    )


@lru_cache(maxsize=None)
def _read_knowledge(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(f"unable to read built-in target knowledge {path}: {error}") from error


def load_knowledge(target: KernelTarget) -> str:
    return "\n\n".join(_read_knowledge(path) for path in knowledge_paths(target))
