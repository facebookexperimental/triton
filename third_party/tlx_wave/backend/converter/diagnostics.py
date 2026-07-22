"""Structured diagnostics for the TLX Wave converter."""

from dataclasses import dataclass


@dataclass
class Diagnostic(Exception):
    code: str
    stage: str
    reason: str
    source_op_index: int | None = None
    source_value_id: int | None = None
    target_op_id: int | None = None
    target_value_id: int | None = None
    fact_id: int | None = None
    layout_map_id: int | None = None
    no_fallback: bool = True

    def __str__(self):
        parts = [f"{self.code} [{self.stage}]: {self.reason}"]
        if self.source_op_index is not None:
            parts.append(f"source_op={self.source_op_index}")
        if self.source_value_id is not None:
            parts.append(f"source_value={self.source_value_id}")
        if self.target_op_id is not None:
            parts.append(f"target_op={self.target_op_id}")
        if self.target_value_id is not None:
            parts.append(f"target_value={self.target_value_id}")
        if self.fact_id is not None:
            parts.append(f"fact={self.fact_id}")
        if self.layout_map_id is not None:
            parts.append(f"layout_map={self.layout_map_id}")
        if self.no_fallback:
            parts.append("no_fallback")
        return "; ".join(parts)


def fail(
    code: str,
    stage: str,
    reason: str,
    *,
    source_op_index: int | None = None,
    source_value_id: int | None = None,
    target_op_id: int | None = None,
    target_value_id: int | None = None,
    fact_id: int | None = None,
    layout_map_id: int | None = None,
):
    raise Diagnostic(
        code,
        stage,
        reason,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
        target_op_id=target_op_id,
        target_value_id=target_value_id,
        fact_id=fact_id,
        layout_map_id=layout_map_id,
    )
