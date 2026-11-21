"""Paper-faithful handoff from the paper's IR to an expert CUDA implementation."""

from ._schema import ArtifactError, AuditError, SKCError
from .compiler import (
    audit_authoring,
    audit_bundle,
    audit_mapping,
    audit_memory,
    audit_sync,
    AuditResult,
    HandoffResult,
    prepare_manual_cuda_handoff,
    scaffold_manual_cuda,
    ScaffoldResult,
)

__all__ = [
    "ArtifactError",
    "AuditError",
    "AuditResult",
    "HandoffResult",
    "SKCError",
    "ScaffoldResult",
    "audit_authoring",
    "audit_bundle",
    "audit_mapping",
    "audit_memory",
    "audit_sync",
    "prepare_manual_cuda_handoff",
    "scaffold_manual_cuda",
]
