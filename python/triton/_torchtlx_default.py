"""How torchTLX engages when nothing overrides it.

Read by ``torch._inductor.config.tlx_mode_default``, which imports this module
directly rather than anything under ``triton.language.extra.tlx``: that
package eagerly pulls in the whole TLX DSL, and this value is needed in every
Inductor process, including ones that never touch a GPU.

``TORCHINDUCTOR_TLX_MODE`` and, internally, an integer JustKnob both take
precedence. Deliberately allowed to differ between Triton builds -- a
conservative build ships ``None`` (off), a fast-moving one ships ``"allow"``.
Keep that divergence when porting this file between builds.
"""

from typing import Literal

DEFAULT_MODE: Literal["allow", "force"] | None = "allow"
