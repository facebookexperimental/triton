"""Schema-only source program records for the TLX Wave converter."""

from dataclasses import dataclass, field


TLX_WAVE_ENABLE_SPLIT_BARRIERS_ATTR = "tlx_wave.enable_split_barriers"
TLX_WAVE_ENABLE_MULTI_WAVE_SPECIALIZATION_ATTR = (
    "tlx_wave.enable_multi_wave_specialization"
)


@dataclass(frozen=True)
class SourceType:
    raw: str
    kind: str
    shape: tuple[int, ...] = ()
    element_type: str | None = None
    element_byte_width: int | None = None
    pointee_type: str | None = None
    encoding: str | None = None
    encoding_attr: object | None = None
    memory_space: str | None = None
    mutable: bool | None = None
    alloc_shape: tuple[int, ...] = ()
    address_space: int | None = None
    pointer_range: int | None = None
    divisibility: int | None = None


@dataclass(frozen=True)
class SourceValue:
    value_id: int
    type: SourceType
    owner_op_index: int | None = None
    producer_name: str | None = None
    argument_index: int | None = None
    region_id: int | None = None
    region_arg_index: int | None = None


@dataclass(frozen=True)
class SourceOp:
    index: int
    name: str
    operands: tuple[int, ...] = ()
    results: tuple[int, ...] = ()
    attrs: dict = field(default_factory=dict)
    region_ids: tuple[int, ...] = ()
    parent_region_id: int | None = None
    parent_op_index: int | None = None
    region_index: int | None = None


@dataclass(frozen=True)
class SourceRegion:
    region_id: int
    op_indices: tuple[int, ...]
    block_arg_ids: tuple[int, ...] = ()
    parent_op_index: int | None = None
    region_index: int | None = None


@dataclass(frozen=True)
class KernelInfo:
    name: str
    target: str | None = None
    num_ctas: int | None = None
    num_warps: int | None = None
    threads_per_warp: int | None = None
    noinline: bool | None = None
    arg_ids: tuple[int, ...] = ()
    enable_split_barriers: bool = False
    enable_multi_wave_specialization: bool = False


@dataclass(frozen=True)
class SourceProgram:
    kernel: KernelInfo
    ops: tuple[SourceOp, ...]
    values: dict[int, SourceValue]
    regions: tuple[SourceRegion, ...]
    top_region_id: int
