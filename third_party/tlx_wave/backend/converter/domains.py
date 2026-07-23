"""Lowering-domain ownership for the staged TLX Wave converter."""

from dataclasses import dataclass

STAGE = "domains"


@dataclass(frozen=True)
class LoweringDomain:
    name: str
    source_ops: tuple[str, ...]
    target_ops: tuple[str, ...]


LOWERING_DOMAINS = (
    LoweringDomain(
        "arithmetic_control",
        (
            "arith.constant",
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.divsi",
            "arith.divui",
            "arith.remsi",
            "arith.remui",
            "arith.addf",
            "arith.divf",
            "arith.maximumf",
            "arith.maxnumf",
            "arith.subf",
            "arith.mulf",
            "arith.extf",
            "arith.cmpi",
            "arith.maxsi",
            "arith.minsi",
            "arith.index_cast",
            "arith.select",
            "llvm.intr.assume",
            "math.exp2",
            "tt.reduce",
            "tt.make_range",
            "tt.splat",
            "tt.addptr",
            "tt.broadcast",
            "tt.expand_dims",
            "tt.join",
            "tt.split",
            "tt.reshape",
            "tt.trans",
            "tt.get_program_id",
            "gpu.thread_id",
            "rocdl.workitem.id.x",
            "ttg.barrier",
            "rocdl.s.barrier",
            "amdg.cond_barrier",
            "rocdl.s.setprio",
            "rocdl.sched.barrier",
            "scf.for",
            "scf.if",
            "tt.return",
        ),
        (
            "constant",
            "affine_materialize",
            "type_convert",
            "binary",
            "float_binary",
            "float_unary",
            "float_cast",
            "cmpi",
            "cmpi_select",
            "maxsi",
            "minsi",
            "assume",
            "make_range",
            "splat",
            "broadcast",
            "make_buffer",
            "addptr",
            "expand_dims",
            "component_join",
            "component_split",
            "program_id",
            "thread_id",
            "barrier",
            "cond_barrier",
            "set_priority",
            "sched_barrier",
            "for_loop",
            "if",
            "select",
            "reduction",
            "return",
        ),
    ),
    LoweringDomain(
        "memory_dma",
        (
            "amdg.buffer_load_to_local",
            "amdg.buffer_load",
            "ttg.async_commit_group",
            "ttg.async_wait",
        ),
        (
            "buffer_load_to_local",
            "buffer_load",
            "token",
            "token_join",
            "issue_token",
            "async_commit_group",
            "async_wait",
        ),
    ),
    LoweringDomain(
        "generic_memory",
        (
            "tt.load",
            "tt.store",
        ),
        (
            "load",
            "store",
        ),
    ),
    LoweringDomain(
        "local_memory_layout",
        (
            "ttg.local_alloc",
            "ttg.memdesc_index",
            "ttg.memdesc_reinterpret",
            "ttg.memdesc_reshape",
            "ttg.memdesc_subslice",
            "ttg.memdesc_trans",
            "ttg.local_load",
            "ttg.local_store",
            "ttg.convert_layout",
        ),
        (
            "local_alloc",
            "memdesc_index",
            "memdesc_view",
            "local_load",
            "local_store",
            "local_load_mma_payload",
            "layout_convert",
        ),
    ),
    LoweringDomain(
        "mfma_fragment",
        (
            "arith.constant",
            "arith.truncf",
            "tt.dot",
            "tt.dot_scaled",
        ),
        (
            "mma_packet_constant",
            "mma",
            "mma_scaled",
            "mma_packet_truncf",
        ),
    ),
    LoweringDomain(
        "store_epilogue",
        ("amdg.buffer_store", ),
        ("buffer_store", ),
    ),
)

DOMAIN_NAMES = tuple(domain.name for domain in LOWERING_DOMAINS)


def source_domains_for_op(op_name):
    return tuple(domain.name for domain in LOWERING_DOMAINS if op_name in domain.source_ops)


def target_domain_for_op(op_kind):
    for domain in LOWERING_DOMAINS:
        if op_kind in domain.target_ops:
            return domain.name
    return None


def all_source_ops():
    return frozenset(op_name for domain in LOWERING_DOMAINS for op_name in domain.source_ops)


def all_target_ops():
    return frozenset(op_kind for domain in LOWERING_DOMAINS for op_kind in domain.target_ops)
