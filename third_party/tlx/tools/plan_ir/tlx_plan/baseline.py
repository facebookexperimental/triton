"""M1.1 fixed FA-backward case catalog and artifact capture helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .model import BaselineManifest, KernelCase, ScheduleSpec, canonical_json


FA_BWD_D128_CASES: dict[str, KernelCase] = {
    "mha_n2048_d128": KernelCase(
        name="mha_n2048_d128",
        family="fa_backward",
        shape={"batch": 1, "q_heads": 1, "kv_heads": 1, "seq_len": 2048, "head_dim": 128},
        dtype="bf16",
        causal=False,
        grouped_query=False,
    ),
    **{
        f"gqa_b16_hq64_hkv8_n{seq_len}_d128": KernelCase(
            name=f"gqa_b16_hq64_hkv8_n{seq_len}_d128",
            family="fa_backward",
            shape={
                "batch": 16,
                "q_heads": 64,
                "kv_heads": 8,
                "seq_len": seq_len,
                "head_dim": 128,
            },
            dtype="bf16",
            causal=False,
            grouped_query=True,
        )
        for seq_len in (1024, 2048, 4096, 8192, 16384)
    },
}


FA_BWD_D128_SCHEDULES: dict[str, ScheduleSpec] = {
    "triton_fused_bf16": ScheduleSpec(
        name="triton_fused_bf16",
        kernel="_attn_bwd_dkdv_dq_d128_triton_fused_kernel",
        implementation="plain_triton",
        config={"block_m": 32, "block_n": 128, "num_warps": 4, "num_stages": 2, "dq_subtile": 2},
        accumulation="bf16",
        algorithm="short_persistent_fused",
    ),
    "tlx_fused_bridge": ScheduleSpec(
        name="tlx_fused_bridge",
        kernel="_attn_bwd_dkdv_dq_d128_gqa_kernel",
        implementation="tlx",
        config={"block_m": 16, "block_n": 256, "num_warps": 4, "num_stages": 1},
        accumulation="mixed_fp32",
        algorithm="long_fused_bridge",
    ),
}


def sha256_file(path: Path) -> str:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{value}"


def make_manifest(
    case_name: str,
    schedule_name: str,
    source_revision: str,
    compiler_revision: str,
    *,
    device: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
    measurements: dict[str, float] | None = None,
) -> BaselineManifest:
    return BaselineManifest(
        case=FA_BWD_D128_CASES[case_name],
        schedule=FA_BWD_D128_SCHEDULES[schedule_name],
        source_revision=source_revision,
        compiler_revision=compiler_revision,
        device=device or {},
        artifacts=artifacts or {},
        measurements=measurements or {},
    )


def capture_compiled_artifacts(
    asm: Mapping[str, Any], output_dir: Path, *, stems: tuple[str, ...] = ("ttgir", "llir", "amdgcn")
) -> dict[str, str]:
    """Save textual compiler artifacts and return content-addressed references."""
    output_dir.mkdir(parents=True, exist_ok=True)
    references: dict[str, str] = {}
    for name in stems:
        value = asm.get(name)
        if value is None:
            continue
        path = output_dir / f"kernel.{name}"
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(str(value))
        references[name] = f"{path.name}@{sha256_file(path)}"
    return references


def write_catalog(path: Path) -> None:
    value = {
        "cases": {name: case.__dict__ for name, case in FA_BWD_D128_CASES.items()},
        "schedules": {name: schedule.__dict__ for name, schedule in FA_BWD_D128_SCHEDULES.items()},
    }
    path.write_text(canonical_json(value))


def read_manifest(path: Path) -> BaselineManifest:
    return BaselineManifest.from_dict(json.loads(path.read_text()))
