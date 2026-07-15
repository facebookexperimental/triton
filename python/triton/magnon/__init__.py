"""compile_iq — decoupled, PTX-direct ptxas-ACF tuning for Triton kernels.

Three stages, communicating only through disk (zero user surface):
  1. collection : a gated hook in jit.py dumps a SOURCE-FREE task (kernel.ptx + spec.json) per kernel.
  2. factory    : offline search tunes an ACF per task -> ACF store. It NEVER recompiles from source;
                  it only runs ptxas (PTX->SASS) and launches the cubin via the CUDA driver API. The
                  search engine (CompileIQ) and its search space are bring-your-own, provisioned
                  separately; this package provides the PTX-direct launch/scoring bridge.
  3. consumption: a gated hook (apply_compile_iq_acf, via core _maybe_apply_compile_iq) re-assembles
                  the cubin from the (cached or fresh) PTX with --apply-controls for a stored ACF and
                  stashes it as a candidate; the first launch A/B-benchmarks plain vs ACF and keeps the
                  winner (never a regression). All in-memory -- Triton's compile cache stays plain, so
                  no TRITON_ALWAYS_COMPILE is needed (a cache hit still re-checks the ACF store).

Enable collection by setting TRITON_COMPILE_IQ_COLLECT=1; no kernel/Triton-user code changes are
required. Submodules (collector / consume / store / ptx_launch) are imported directly where used;
this package marker stays empty on purpose.
"""
