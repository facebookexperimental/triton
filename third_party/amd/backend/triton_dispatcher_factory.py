# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Factory for the meta-specific HIP ``_TritonDispatcher``."""

from triton.runtime.driver import driver

_bridge_registered = False


def _load_module():
    global _bridge_registered
    utils = driver.active.utils
    if not _bridge_registered:
        _bridge_registered = True
        try:
            from triton._C._torch_bridge import get_tensor_access_capsule
            utils.register_tensor_bridge(get_tensor_access_capsule())
        except (ImportError, AttributeError):
            pass
    return utils


def make_triton_dispatcher(schema, function: int):
    """Bind a scalar/pointer HIP kernel's invariant launch metadata once.

    Tensor descriptors and compiler-managed scratch still use HIPLauncher.  A
    ``None`` return is the explicit signal for CompiledKernel to take that
    existing, fully general path.
    """
    if schema.get("global_scratch_size", 0) or schema.get("profile_scratch_size", 0):
        return None
    if schema.get("tensordesc_meta"):
        return None

    arg_types = []
    for arg in schema["args"]:
        ty = arg["type"]
        # Structured arguments require the generic launcher's recursive
        # flattening. Device tensor descriptors require TDM construction.
        if "(" in ty or ty.startswith("tensordesc"):
            return None
        arg_types.append(ty)

    utils = _load_module()
    type_codes = tuple(utils.build_signature_metadata(arg_types))
    return utils._TritonDispatcher(
        function=function,
        num_warps=schema["num_warps"],
        num_ctas=schema["num_ctas"],
        shared_mem=schema["shared_mem"],
        launch_cooperative_grid=1 if schema.get("launch_cooperative_grid", False) else 0,
        warp_size=schema.get("warp_size",
                             driver.active.get_current_target().warp_size),
        arg_type_codes=type_codes,
    )
