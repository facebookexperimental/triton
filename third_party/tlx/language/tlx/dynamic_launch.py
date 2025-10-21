# from dataclasses import dataclass
import triton.language.core as tl
# from triton.language.core import _aggregate as aggregate

from . import types as tlx
# from . import barrier as tlx_barrier

# Blackwell-only


@tl.builtin
def alloc_clc_responses(
    num_responses: tl.constexpr,
    _semantic=None,
) -> tlx.clc_response:
    layout = tlx.swizzled_shared_layout_encoding.make_default(rank=1)
    layout_handle = _semantic.builder.make_swizzled_shared_encoding_attr(
        layout.vectorSize,
        layout.perPhase,
        layout.maxPhase,
        layout.order,
        layout.numCTAsPerCGA,
        layout.numCTASplit,
        layout.numCTAOrder,
    )
    return tlx.clc_response(_semantic.builder.create_alloc_clc_responses(num_responses.value, layout_handle),
                            num_responses, layout, _semantic)


@tl.builtin
def clc_issue(
    clc_response_addr: tlx.clc_response,
    barrier: tlx.mbarrier,
    _semantic=None,
):
    # Issue async `clusterlaunchcontrol.try_cancel` request for
    # CTA ID of available cluster
    assert isinstance(clc_response_addr, tlx.clc_response)

    return _semantic.builder.clc_issue(clc_response_addr.handle, barrier.handle)


@tl.builtin
def clc_query(
    clc_response_addr: tlx.clc_response,
    _semantic=None,
):
    # Extract CTA ID from CLC response
    assert isinstance(clc_response_addr, tlx.clc_response)

    x = _semantic.builder.clc_query(clc_response_addr.handle, )
    return _semantic.tensor(x, tl.int32)


# @dataclass
# class pipeline_state:
#     _stages: int
#     _index: int = 0
#     _phase: int = 0
#     _count: int = 0

#     def incr(self)   -> None:
#         self._index += 1
#         self._count += 1
#         if self._index == self._stages:
#             self._index = 0
#             self._phase ^= 1


# @aggregate
# class pipeline_clc_fetch_async:
#     full_bars: tlx.mbarrier
#     empty_bars: tlx.mbarrier
#     responses: tlx.clc_response
#     _num_stages: int = 0
#     _state: pipeline_state

#     @tl.builtin
#     def __init__(self, num_stages, _semantic=None) -> None:
#         # self._semantic = _semantic
#         self._num_stages = tl._unwrap_if_constexpr(num_stages)
#         self._state = pipeline_state(self._num_stages)
#         self.full_bars = tlx_barrier.alloc_barriers(num_barriers=self._num_stages)
#         self.empty_bars = tlx_barrier.alloc_barriers(num_barriers=self._num_stages)
#         self.responses = alloc_clc_responses(num_responses=self._num_stages)

#     # @triton.jit
#     # def alloc(self) -> None:
#     #     # alloc mbar and clc responses        
#     #     full_bars = tlx.alloc_barriers(num_barriers=self._num_stages)
#     #     empty_bars = tlx.alloc_barriers(num_barriers=self._num_stages)
#     #     responses = tlx.alloc_clc_responses(num_responses=self._num_stages)

#     # @triton.jit
#     def fetch_next_work(self) -> int:
#         clc_mbar = self.full_bars[0]
#         clc_response = self.responses[0]

#         # TLX
#         # Issue async clc.try_cancel for the next available CTA
#         tlx.barrier_expect_bytes(clc_mbar, 16)  # CLC response is 16-byte
#         tlx.clc_issue(clc_response, clc_mbar)

#         # Wait for clc.try_cancel finishes
#         tlx.barrier_wait(clc_mbar, self._phase)
#         self._phase = self._phase ^ 1

#         # Extract CTA ID from CLC response
#         tile_id = tlx.clc_query(clc_response)

#         return tile_id
