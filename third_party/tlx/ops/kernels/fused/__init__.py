"""Kernels that fuse steps which would otherwise run as separate launches: an elementwise epilogue into a GEMM
(``addmm_mul_add``, ``linear_rowscale``), the norms and residual update of a residual-stream boundary
(``norm_residual_norm``), or a per-head norm and its output gate (``rmsnorm_gate``)."""
