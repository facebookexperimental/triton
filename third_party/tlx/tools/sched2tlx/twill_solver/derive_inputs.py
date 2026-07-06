"""Derive the three Twill inputs the modulo scheduler never models:
``regs(v)`` (per-thread register footprint), ``blocking(u,v)`` (edge needs a
blocking sync to consume an async result), and ``variable_latency(v)`` /
streaming (variable-latency op with no incoming data dependence, e.g. a TMA
input load). All are computed from data already in the schedule/DDG graph.
"""

from __future__ import annotations

import re

from .ddg_model import LoopModel, SolverNode

# Op kinds whose result lives in the register file of the owning warp group
# (a distributed tensor). SMEM/TMEM/async/scalar results cost ~0 registers.
_REGISTER_RESIDENT_KINDS = {
    "ttng.tmem_load", "ttg.local_load", "ttg.convert_layout", "tt.reduce",
    "tt.broadcast", "tt.expand_dims", "tt.splat", "tt.reshape", "tt.trans",
    "tt.join", "tt.split", "math.exp2", "math.exp", "math.log2", "math.log",
    "math.sqrt", "math.rsqrt", "math.tanh",
    "arith.mulf", "arith.addf", "arith.subf", "arith.divf", "arith.maxnumf",
    "arith.minnumf", "arith.truncf", "arith.extf", "arith.cmpf",
    "arith.select", "arith.fptosi", "arith.sitofp",
}

_TENSOR_RE = re.compile(r"tensor<([0-9x]+)x([a-z0-9]+)\b")
_BITS = {"f32": 32, "f16": 16, "bf16": 16, "i32": 32, "i16": 16, "i8": 8,
         "f8": 8, "i1": 8, "i64": 64, "f64": 64}
BYTES_PER_REG = 4
THREADS_PER_WARP = 32


def _tensor_shape_bits(result_type: str) -> tuple[int, int] | None:
    """Parse ``tensor<128x64xf16, ...>`` -> (num_elements, elem_bits)."""
    m = _TENSOR_RE.search(result_type)
    if not m:
        return None
    dims = m.group(1).split("x")
    try:
        n = 1
        for d in dims:
            n *= int(d)
    except ValueError:
        return None
    bits = _BITS.get(m.group(2), 32)
    return n, bits


def regs(node: SolverNode, num_warps: int) -> int:
    """Per-thread register footprint of holding ``node``'s result live in a warp
    group of ``num_warps`` warps. 0 for async / memory / scalar results.
    """
    if node.op_kind not in _REGISTER_RESIDENT_KINDS:
        return 0
    parsed = _tensor_shape_bits(node.result_type)
    if parsed is None:
        return 0
    n_elems, bits = parsed
    total_bytes = n_elems * bits // 8
    threads = max(1, num_warps) * THREADS_PER_WARP
    return (total_bytes + threads * BYTES_PER_REG - 1) // (threads * BYTES_PER_REG)


def blocking(loop: LoopModel, src: int, dst: int) -> bool:
    """Does consuming edge (src->dst) require a blocking sync? True when the
    producer emits an asynchronous result (TMA load via mbarrier, TC MMA via
    TMEM) that the consumer must wait on before issuing.
    """
    sn = loop.node_by_id(src)
    return sn.pipeline in ("TMA", "TC")


def variable_latency(loop: LoopModel, node: SolverNode) -> bool:
    """Streaming variable-latency op: a TMA load with no incoming intra-iteration
    data dependence (its inputs are loop-invariant descriptors/offsets), so it can
    run ahead of the pipeline. Twill assigns these zero latency + a tunable depth.
    """
    if node.pipeline != "TMA":
        return False
    for e in loop.edges:
        if e.dst == node.id and e.distance == 0:
            return False
    return True
