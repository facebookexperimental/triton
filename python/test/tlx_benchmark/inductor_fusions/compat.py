from __future__ import annotations

import inspect


class CompatibilityError(RuntimeError):
    pass


def require_include_fallback(register_custom_op_autotuning) -> None:
    try:
        parameters = inspect.signature(register_custom_op_autotuning).parameters
    except (TypeError, ValueError) as error:
        raise CompatibilityError(
            "cannot inspect torch._inductor.kernel.custom_op.register_custom_op_autotuning") from error
    if "include_fallback" not in parameters:
        raise CompatibilityError("the installed PyTorch is incompatible: "
                                 "register_custom_op_autotuning(include_fallback=...) is required")


def install_torchtlx(*, require_custom_op_autotuning: bool) -> tuple[object, object]:
    try:
        import torch
        import triton
        from torch._inductor import config
    except ImportError as error:
        raise CompatibilityError("PyTorch, Triton, and TorchInductor must be installed") from error

    if not hasattr(config.triton, "tlx_mode"):
        raise CompatibilityError("the installed PyTorch does not expose torch._inductor.config.triton.tlx_mode")

    if require_custom_op_autotuning:
        try:
            from torch._inductor.kernel.custom_op import register_custom_op_autotuning
        except ImportError as error:
            raise CompatibilityError("the installed PyTorch does not provide register_custom_op_autotuning") from error
        require_include_fallback(register_custom_op_autotuning)

    # PyTorch does not own this loader. Importing the FBTriton registry installs
    # TorchTLX choices and pattern registrations before compilation begins.
    try:
        import triton.language.extra.tlx.inductor.registry  # noqa: F401
    except ImportError as error:
        raise CompatibilityError("the installed Triton is not this FBTriton checkout") from error
    return torch, triton
