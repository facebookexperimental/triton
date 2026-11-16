"""Bind flat HIP launch arguments to the shared native JIT dispatcher."""

from triton.runtime import _allocation
from triton.runtime.driver import driver


class _ProfileAllocator:
    """Preserve the HIP launcher's default profile allocator."""

    def get(self):
        if _allocation.has_profile_allocator():
            return _allocation._profile_allocator.get()
        return driver.active.allocate_default_profile_scratch


_profile_allocator = _ProfileAllocator()


def make_triton_dispatcher(schema, function):
    """Create a HIP dispatcher, or decline unsupported argument signatures."""
    types = [arg["type"] for arg in schema["args"]]
    if any("(" in ty or ty.startswith("tensordesc") for ty in types):
        return None
    cluster = schema.get("ctas_per_cga")
    if cluster is not None and (len(cluster) != 3 or cluster[1:] != [1, 1]):
        return None
    schema = dict(schema)
    if cluster is not None:
        schema["physical_cluster_x"] = cluster[0]
        schema["num_ctas"] = cluster[0]
    utils = driver.active.utils
    signature = utils.build_signature_metadata(types)
    return utils._TritonDispatcher(schema, function, signature, _allocation._allocator, _profile_allocator)
