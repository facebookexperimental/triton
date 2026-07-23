"""driver.install(mode, binding) — patch FA4's interface to use SKC shims.

One binding per process (compile_cache on _flash_attn_fwd/_bwd is keyed
without knowledge of our overrides; the bench worker enforces the
process-per-binding discipline).  FA4-venv only.
"""

import hashlib
import json
import os


def install(mode: str, binding: dict | None):
    """mode in {'fwd', 'bwd'}; binding=None installs the identity shim (M2)."""
    assert mode in ("fwd", "bwd")
    assert os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0") != "1", \
        "disk compile cache must stay off: overridden kernels share compile_keys"

    from . import fa4_pin
    pin = fa4_pin.verify_pin()

    import flash_attn.cute.interface as interface
    if mode == "fwd":
        from .shim_fwd import SKCForwardSm100
        SKCForwardSm100._skc = binding or {}
        SKCForwardSm100._skc_audit = []
        interface.FlashAttentionForwardSm100 = SKCForwardSm100
        shim = SKCForwardSm100
    else:
        from .shim_bwd import SKCBackwardSm100
        SKCBackwardSm100._skc = binding or {}
        SKCBackwardSm100._skc_audit = []
        interface.FlashAttentionBackwardSm100 = SKCBackwardSm100
        shim = SKCBackwardSm100

    audit_hash = hashlib.sha256(
        json.dumps(binding or {}, sort_keys=True).encode()).hexdigest()[:16]
    return {"pin_commit": pin["commit"], "binding_hash": audit_hash,
            "shim": shim.__name__}
