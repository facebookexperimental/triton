"""Logical-relation core for TTGIR reduction equivalence (M1) — MLIR-native.

Given two compiled **TTGIR** modules, decide whether their reductions are
*bitwise-equivalent* — i.e. whether they combine the same elements in the same
floating-point association order. We extract the reduction-order signature by
**inspecting the parsed MLIR** with the compiler's own layout machinery
(``toLinearLayout`` + the ``tt.reduce`` op accessors), exposed from C++ via
``triton._C.libtriton.bitequiv`` (see ``lib/Analysis/ReductionOrder.cpp`` and
``python/src/bitequiv.cc``) — NOT by regex over the IR text.

The same C++ analysis is reusable inside the compiler (a verification pass) to
assert the reduction order is preserved across passes once the ordering switch
(``inner_tree``) is on.

Public API (contract unchanged; consumed by ``bitequiv/equivalence_ttgir.py`` and the
autotuner ``ir_config_prune`` hook). Names mirror the PTX checker
(``bitequiv/ptx_reduction.py``): ``ttgir_reduction_descriptor`` ↔ ``ptx_reduction_descriptor``,
``ttgir_reductions_equivalent`` ↔ ``ptx_reductions_equivalent``:

  * ``ttgir_reduction_descriptor(ttgir) -> tuple[str, ...]`` — one canonical signature per
    ``tt.reduce`` (plus a conservative entry for any MMA/dot accumulation). Equal
    descriptors ⇒ same reduction tree ⇒ same bits (sound). ``()`` only when the
    module has no reduction-like op at all.
  * ``ttgir_reductions_equivalent(a, b) -> bool``.

The human-readable design is in ``bitequiv/CLAUDE.md`` / ``bitequiv/PROGRESS.md``.
This module is pure Python but requires the built ``triton`` (it calls into
``libtriton``); parsing a TTGIR string is CPU-only (no GPU, no kernel launch).

This TTGIR checker is the sibling of the PTX checker (``bitequiv/ptx_reduction.py``,
in-tree on this stack): TTGIR fixes the association order; the PTX checker also
catches FMA contraction, which is decided below TTGIR and is invisible here. Both
are measured by the shared evaluation framework in ``bitequiv/evaluation/`` (run the
TTGIR checker with ``--checker bitequiv.ttgir_reduction:ttgir_reduction_descriptor
--artifact ttgir``).
"""

_ctx = None


def _libtriton():
    from triton._C import libtriton
    return libtriton


def _context():
    """A cached MLIRContext with the TTGIR dialects loaded (core + NVIDIA)."""
    global _ctx
    if _ctx is None:
        lt = _libtriton()
        ctx = lt.ir.context()
        lt.ir.load_dialects(ctx)  # TritonGPU / Triton / TritonNvidiaGPU / arith / ...
        try:
            lt.nvidia.load_dialects(ctx)  # NVGPU / NVWS (warp-spec / TMA TTGIR ops)
        except Exception:  # noqa: BLE001 - non-NVIDIA build: core dialects already cover plain TTGIR
            pass
        _ctx = ctx
    return _ctx


def ttgir_reduction_descriptor(ttgir):
    """Canonical, hashable reduction-order descriptor of a TTGIR module: a tuple of
    per-``tt.reduce`` signature strings built by the MLIR-native C++ analysis (one
    per reduce, plus a conservative entry for any MMA/dot accumulation). Two TTGIRs
    with equal descriptors perform identical reduction trees (sound). ``()`` when
    the module has no reduction-like op."""
    if not ttgir:
        return ()
    return tuple(_libtriton().bitequiv.reduction_order_signatures(ttgir, _context()))


def ttgir_reductions_equivalent(ttgir_a, ttgir_b):
    """True iff the two TTGIRs reduce in the same (bitwise-equivalent) order,
    decided statically from the parsed MLIR (no kernel launch, no inputs)."""
    return ttgir_reduction_descriptor(ttgir_a) == ttgir_reduction_descriptor(ttgir_b)
