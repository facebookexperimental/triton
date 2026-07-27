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


def _fma_eligible(ttgir):
    """True iff the module has a float multiply that ``enable_fp_fusion`` could contract
    into an adjacent add — a ``dot``/``warp_group_dot`` accumulation or a plain
    ``arith.mulf``. A pure-add reduction has no such multiply, so its bits are invariant to
    fp fusion and must NOT be split on it (that would over-split). Deliberately a broad
    text test: a ``mulf`` that does not actually feed an add only causes an always-sound
    over-split, never an over-merge."""
    return "dot" in ttgir or "arith.mulf" in ttgir


def ttgir_reduction_descriptor(ttgir, config=None):
    """Canonical, hashable reduction-order descriptor of a TTGIR module: a tuple of
    per-``tt.reduce`` signature strings built by the MLIR-native C++ analysis (one
    per reduce, plus a conservative entry for any MMA/dot accumulation). Two TTGIRs
    with equal descriptors perform identical reduction trees (sound). ``()`` when
    the module has no reduction-like op.

    ``config`` (optional autotuner config): FMA contraction is decided BELOW TTGIR (ptxas
    fuses ``mul``+``add`` into ``fma`` per ``enable_fp_fusion``), so it is invisible in the
    IR — two configs that differ only in ``enable_fp_fusion`` compile to the SAME TTGIR yet
    produce DIFFERENT bits. When a config is supplied and the module is fma-eligible, its
    ``enable_fp_fusion`` value is folded into the descriptor so those configs split (the M1
    ``dot`` reduction and the GEMM epilogue). The PTX checker reads fusion from the PTX
    directly and is called without this arg, so it is unaffected."""
    if not ttgir:
        return ()
    sigs = tuple(_libtriton().bitequiv.reduction_order_signatures(ttgir, _context()))
    if config is not None and sigs:
        fp_fusion = getattr(config, "enable_fp_fusion", None)
        if fp_fusion is not None and _fma_eligible(ttgir):
            sigs = sigs + (f"enable_fp_fusion={fp_fusion}", )
    return sigs


def ttgir_reductions_equivalent(ttgir_a, ttgir_b, config_a=None, config_b=None):
    """True iff the two TTGIRs reduce in the same (bitwise-equivalent) order,
    decided statically from the parsed MLIR (no kernel launch, no inputs). Pass the
    per-module configs to also decide the fp-fusion split (see ``ttgir_reduction_descriptor``)."""
    return ttgir_reduction_descriptor(ttgir_a, config_a) == ttgir_reduction_descriptor(ttgir_b, config_b)
