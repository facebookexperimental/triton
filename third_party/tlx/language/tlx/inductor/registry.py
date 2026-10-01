"""Import gate for the torchTLX inductor integration.

Importing the integration registers TLX template heuristics, swaps
``config.inductor_choices_class``, and monkey-patches ``TritonTemplate`` /
``TritonTemplateKernel`` -- work that regressed default-path compile time
(S713810) for processes that never engage TLX. This module therefore loads the
implementation (``registry_impl``) eagerly only when ``torch_tlx_enabled()``
holds at first import. A bare ``import registry`` otherwise stays cheap and
exposes just ``tlx_only_cuda_options`` -- the single shared list object from
``tlx_config`` (``[]`` unless enabled) -- so the ``torch._inductor.utils``
fallback import keeps working.

Explicit use of any other name loads the implementation on demand via
``__getattr__`` (idempotent through ``sys.modules``), preserving the pre-gate
contract for direct consumers such as the template unit tests. Asking for an
implementation name is opt-in; processes that never touch one pay nothing.
"""

from typing import Any

from .tlx_config import torch_tlx_enabled
from .tlx_config import tlx_only_cuda_options as tlx_only_cuda_options

__all__ = ["tlx_only_cuda_options", "torch_tlx_enabled"]

if torch_tlx_enabled():
    # Eager side-effect import: registers heuristics and installs the
    # TritonTemplateKernel patches. Required at import time because torch's
    # maybe_install() relies on these side effects having happened.
    from .registry_impl import get_heuristic_config as get_heuristic_config

    __all__ = ["get_heuristic_config", *__all__]


def __getattr__(name: str) -> Any:
    from . import registry_impl

    return getattr(registry_impl, name)
