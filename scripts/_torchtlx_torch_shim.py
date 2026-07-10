#!/usr/bin/env python3
"""Idempotently add the torchTLX PyTorch-side integration to the active torch.

torchTLX needs two things in ``torch._inductor`` that current OSS nightly
PyTorch does not yet ship:

  1. the ``tlx_mode`` knob on ``torch._inductor.config.triton``
  2. a ``template_heuristics/tlx.py`` that imports the fbtriton fork integration
     (``triton.language.extra.tlx.inductor.registry``) rather than the old
     fbcode-only path.

Both are OSS-bound PyTorch changes. This is a STOPGAP so the torchTLX unit tests
can run against a stock nightly torch; once PyTorch lands them upstream, this
becomes a no-op (it detects the pieces are already present and does nothing).
"""
import os
import sys

try:
    import torch
except Exception as e:  # pragma: no cover
    sys.exit(f"[torch-shim] torch not importable: {e}")

base = os.path.dirname(torch.__file__)
cfg_path = os.path.join(base, "_inductor", "config.py")
tlx_path = os.path.join(base, "_inductor", "template_heuristics", "tlx.py")
changed = False

# 1) config.triton.tlx_mode -----------------------------------------------------
ANCHOR = 'class triton:\n    """\n    Config specific to codegen/triton.py\n    """\n'
INSERT = ('def tlx_mode_from_env() -> "Literal[\'allow\', \'force\'] | None":\n'
          "    # torchTLX shim: only 'allow'/'force' enable TLX; anything else -> None.\n"
          '    mode = os.environ.get("TORCHINDUCTOR_TLX_MODE")\n'
          '    if mode in ("allow", "force"):\n'
          "        return cast(\"Literal['allow', 'force']\", mode)\n"
          "    return None\n\n\n"
          'class triton:\n    """\n    Config specific to codegen/triton.py\n    """\n\n'
          "    # torchTLX enablement (stopgap added by scripts/_torchtlx_torch_shim.py).\n"
          "    tlx_mode: \"Literal['allow', 'force'] | None\" = tlx_mode_from_env()\n")

with open(cfg_path) as f:
    cfg = f.read()

if "tlx_mode" in cfg:
    print("[torch-shim] config.triton.tlx_mode already present")
elif ANCHOR not in cfg:
    sys.exit("[torch-shim] could not find the 'class triton:' anchor in config.py "
             "(torch layout changed) -- update this shim")
elif not ("cast" in cfg and "Literal" in cfg):
    sys.exit("[torch-shim] config.py is missing 'cast'/'Literal' imports -- update this shim")
else:
    with open(cfg_path, "w") as f:
        f.write(cfg.replace(ANCHOR, INSERT, 1))
    changed = True
    print("[torch-shim] added config.triton.tlx_mode")

# 2) template_heuristics/tlx.py -> load the fork -------------------------------
TLX_BODY = ("# Stopgap (scripts/_torchtlx_torch_shim.py): load torchTLX from the fbtriton\n"
            "# fork. Succeeds only when the active Triton is the fork; no-op otherwise.\n"
            "try:\n"
            "    import triton.language.extra.tlx.inductor.registry  # noqa: F401\n"
            "except ImportError:\n"
            "    pass\n")
os.makedirs(os.path.dirname(tlx_path), exist_ok=True)
existing = ""
if os.path.exists(tlx_path):
    with open(tlx_path) as f:
        existing = f.read()
if "triton.language.extra.tlx" in existing:
    print("[torch-shim] template_heuristics/tlx.py already loads the fork")
else:
    with open(tlx_path, "w") as f:
        f.write(TLX_BODY)
    changed = True
    print("[torch-shim] wrote template_heuristics/tlx.py (loads the fork)")

print("[torch-shim] done" + (" (modified torch)" if changed else " (no changes)"))
