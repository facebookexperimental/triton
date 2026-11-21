"""Compatibility imports for the paper-faithful manual CUDA workflow."""

from paper_joint_solver.pipelined_ir import (
    HandoffResult,
    prepare_manual_cuda_handoff,
)

from .audit import (
    audit_authoring,
    audit_bundle,
    audit_mapping,
    audit_memory,
    audit_sync,
    AuditResult,
)
from .scaffold import scaffold_manual_cuda, ScaffoldResult

__all__ = [
    "AuditResult",
    "HandoffResult",
    "ScaffoldResult",
    "audit_authoring",
    "audit_bundle",
    "audit_mapping",
    "audit_memory",
    "audit_sync",
    "prepare_manual_cuda_handoff",
    "scaffold_manual_cuda",
]
