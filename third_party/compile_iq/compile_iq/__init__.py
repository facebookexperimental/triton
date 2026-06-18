"""compile_iq — decoupled ptxas-ACF tuning for Triton kernels.

Three stages, communicating only through disk (zero user surface):
  1. collection : a gated hook in jit.py dumps a "compileIQ task" per kernel.
  2. factory    : offline binary tunes an ACF per task -> ACF store.
  3. consumption: a ptxas shim applies the stored ACF during PTX->SASS (later).

Enable collection by setting the env var FBTRITON_COMPILE_IQ_COLLECT=1. No kernel or
Triton-user code changes are required. Submodules (collector / factory / replay / consume /
store) are imported directly where used; this package marker stays empty on purpose.
"""
