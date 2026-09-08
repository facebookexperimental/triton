"""Perf guardrail for `tlx.ops.hstu_attn_dev`, against production Triton HSTU.

There is no vendor library for SiLU-scaled ragged attention, so the reference is
`_reference.py::triton_hstu_mha` -- the kernel that ships today, vendored into
the op package. "Faster than what production runs" is the question that matters
here, and it is the only one that can be asked.

The two are not tuned symmetrically: the reference autotunes its full space
while TLX runs at whatever `--space` says. That is recorded per case as
`ref_autotuned` rather than corrected for, because forcing the reference onto
one config would measure a kernel nobody runs.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import torch

from triton.tlx.ops.kernels.hstu_attn._shapes import flops, inputs, label

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "hstu_attn_dev"
REF_NAME = "triton.tlx.ops.kernels.hstu_attn._reference:triton_hstu_mha"
#: No `heuristic_config` yet; see `bench_flash_attn.py`.
DEFAULT_SPACE = "full"
EXTRA_COLUMNS = (("tokens", "tokens"), )
#: One fresh-cache first call per direction, not per case: the cold pass
#: compiles this op's whole autotune space, and "does a first call take too
#: long" is not a per-shape question.
COLD_COMPILE = "first"

COMPILE_CAP_S = 900.0

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
DIRECTIONS = ("fwd", "bwd")

REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}


def shapes(synthetic: bool = False) -> list[list]:
    if synthetic:
        from triton.tlx.ops.kernels.hstu_attn._shapes import SYNTHETIC

        return list(SYNTHETIC)
    return list(importlib.import_module(f"triton.tlx.ops.kernels.hstu_attn.{driver.arch()}").PERF_SHAPES)


def cases(synthetic: bool = False) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry[5]]).removeprefix("torch."), shape=tuple(entry[:5]),
             direction=direction, label=label(*entry, direction))
        for entry in shapes(synthetic)
        for direction in DIRECTIONS
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import hstu_attn_dev as tlx_hstu_attn
    from triton.tlx.ops.kernels.hstu_attn._reference import triton_hstu_mha

    Z, max_seq_len, H, head_dim, causal = case.shape
    backward = case.direction == "bwd"
    dtype = getattr(torch, case.dtype)
    q, k, v, offsets, attn_scale = inputs(Z, max_seq_len, H, head_dim, dtype, requires_grad=backward)
    alpha = 1.0 / head_dim
    tokens = int(offsets[-1])

    tlx_fwd = lambda: tlx_hstu_attn(  # noqa: E731
        q, k, v, offsets, max_seq_len, attn_scale, alpha=alpha, causal=causal, arch=driver.arch(), space=space)
    ref_fwd = lambda: triton_hstu_mha(  # noqa: E731
        max_seq_len, alpha, q, k, v, offsets, attn_scale)
    # `tokens` is what a ragged batch actually carried, which the label's
    # Z x MAX_SEQ_LEN only equals under the uniform draw.
    extra = {"tokens": tokens, "ref_autotuned": True}
    flop_count = flops(*case.shape, case.direction, tokens=tokens)

    if not backward:
        return Prepared(tlx_fn=tlx_fwd, ref_fn=ref_fwd, flop_count=flop_count,
                        check=lambda: close_enough(tlx_fwd(), ref_fwd(), REL_PRECISION[case.dtype]), extra=extra,
                        cap_s=COMPILE_CAP_S)

    tlx_out, ref_out = tlx_fwd(), ref_fwd()
    do = torch.randn_like(tlx_out)
    return Prepared(
        tlx_fn=lambda: tlx_out.backward(do, retain_graph=True),
        ref_fn=lambda: ref_out.backward(do, retain_graph=True),
        flop_count=flop_count,
        grad_to_none=[q, k, v],
        check=None,  # see bench_flash_attn.py
        extra=extra,
        cap_s=COMPILE_CAP_S,
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
