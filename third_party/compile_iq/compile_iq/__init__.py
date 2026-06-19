"""compile_iq — decoupled, PTX-direct ptxas-ACF tuning for Triton kernels.

Three stages, communicating only through disk (zero user surface):
  1. collection : a gated hook in jit.py dumps a SOURCE-FREE task (kernel.ptx + spec.json) per kernel.
  2. factory    : offline search tunes an ACF per task -> ACF store. It NEVER recompiles from source;
                  it only runs ptxas (PTX->SASS) and launches the cubin via the CUDA driver API. The
                  reference factory is the EVO route under smoke/ (ptx_evo_search + ptx_bench_one).
  3. consumption: a make_cubin shim appends --apply-controls for a stored ACF during PTX->SASS.

Enable collection by setting TRITON_COMPILE_IQ_COLLECT=1; no kernel/Triton-user code changes are
required. Submodules (collector / consume / store / ptx_launch) are imported directly where used;
this package marker stays empty on purpose.
"""
