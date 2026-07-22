"""Forward abstract interpreter for the PTX bitwise-equivalence checker.

Simulates a kernel FORWARD over a symbolic thread (program order via ``linearize``), maintaining a
per-register value-DAG state, rather than the backward def-use walk in :mod:`bitequiv.ptx.builder`.
This is the migration target from the co-design (``~/bitwise-equiv/bitequiv_interpreter_codesign_3.3.md``):
the forward direction models cross-warp reductions, loops, MMA->reduce->MMA composition, and branches
naturally (a reduction is a forward data flow: produce -> st.shared -> bar -> ld.shared -> combine).

Built incrementally, gated on the empirical fuzzer (0 over-merges) + over-split <= the backward
checker, per phase. Phase 0 (this scaffold): single-block within-thread + within-warp reduction,
reproducing the backward descriptor.
"""
