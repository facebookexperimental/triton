"""SKC Phase B — CuTe-DSL backend via bind-in-place shims over FA4's kernels.

No fork: SKC subclasses FlashAttentionForwardSm100/BackwardSm100, overrides
the bindable parameter surface at two safe write points, and launches through
FA4's own interface.  See SKC_PHASE_B_DESIGN.md.

Package split by venv:
  fa4_pin / binder_cute / tests — pure Python, run in the main venv
  shim_fwd / shim_bwd / driver  — import flash_attn, run only in the FA4 venv
"""
