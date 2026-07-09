"""
TLX-specific codegen for async TMA descriptor store pipeline.

This implements the Blackwell async TMA store pattern that stages values
through SMEM before issuing an async descriptor store. It references
TLX-specific APIs (tlx.local_store, tlx.async_descriptor_store, etc.) and
SMEM buffers (c_smem_buffers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch._inductor.codegen.common import DeferredLine
from torch._inductor.codegen.triton import triton_store_type
from torch._inductor.virtualized import V

if TYPE_CHECKING:
    from torch._inductor.codegen.triton import TritonKernel


def _prepare_store_value(
    kernel: TritonKernel, name: str, indexing: object, value: str
) -> str:
    """Prepare value for async TMA store (broadcast, reshape, cast)."""
    value = f"tl.broadcast_to({value}, {indexing.final_shape})"  # type: ignore[attr-defined]

    for idx, (dim, broadcast_dim) in enumerate(
        zip(indexing.final_shape, indexing.broadcast_shape)  # type: ignore[attr-defined]
    ):
        if V.graph.sizevars.statically_known_equals(dim, broadcast_dim):
            indexing.broadcasting_dims[idx] = False  # type: ignore[attr-defined]

    value = indexing.codegen_broadcast_and_reshape(  # type: ignore[attr-defined]
        value,
        indexing.final_shape,  # type: ignore[attr-defined]
        indexing.block_shape,  # type: ignore[attr-defined]
        allow_implicit=False,
        for_store=True,
    )

    value = f"{value}.to({triton_store_type(V.graph.get_dtype(name))})"
    return value


def codegen_async_tma_store(
    self: TritonKernel, name: str, indexing: object, block_descriptor: str, value: str
) -> None:
    """Generate TLX async TMA descriptor store pipeline.

    Stages the value through SMEM before issuing an async TMA store,
    matching the standalone TLX kernel's store pattern for Blackwell.
    Requires c_smem_buffers and NUM_EPILOGUE_SMEM_BUFFERS to be defined
    in the template.

    The SMEM buffer index can be controlled by setting
    async_tma_store_buf_idx on the kernel to a Triton runtime expression
    string (e.g. "group_id * EPILOGUE_SUBTILE + slice_id"). If not set,
    falls back to a compile-time counter.
    """
    value = _prepare_store_value(self, name, indexing, value)

    buf_idx_expr = getattr(self, "async_tma_store_buf_idx", None)
    if buf_idx_expr is None:
        counter = getattr(self, "_async_tma_store_counter", 0)
        self._async_tma_store_counter = counter + 1  # type: ignore[attr-defined]
        buf_idx_expr = str(counter)

    offsets_str = self.index_to_str(indexing.offsets)  # type: ignore[attr-defined]

    lines = [
        f"c_smem = c_smem_buffers[({buf_idx_expr}) % NUM_EPILOGUE_SMEM_BUFFERS]",
        "tlx.async_descriptor_store_wait(1)",
        f"tlx.local_store(c_smem, {value})",
        "tlx.fence_async_shared()",
        f'tlx.async_descriptor_store({block_descriptor}, c_smem, {offsets_str}, eviction_policy="evict_first")',
    ]
    self._handle_pdl_before_access(self.stores, name, consider_reads=True)
    for line in lines:
        self.stores.writeline(DeferredLine(name, line))

    if not self.inside_reduction:
        self.outside_loop_vars.add(value)
