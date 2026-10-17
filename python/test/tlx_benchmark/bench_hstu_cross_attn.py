"""Perf guardrail for `tlx.ops.hstu_cross_attn`, against the Triton HSTU cross attention.

The reference is the Triton kernel in
`tutorials/hstu_cross_attn/triton_bw_cross_attention.py` (TMA off, as on AMD),
loaded from the tutorial directory of this checkout. Like `hstu_attn`, there is
no vendor library for SiLU-scaled jagged attention, and the softmax shapes are
jagged cross attention with independent q and KV offsets, so "faster than what
production runs" is the comparison that matters.

The reference autotunes its full space; TLX runs at whatever `--space` says.
FLOPs count QK^T and PV over the actual (unpadded) lengths; backward is 2.5x.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import torch

from triton.tlx.ops.kernels.hstu_cross_attn._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.hstu_cross_attn._shapes import SYNTHETIC, flops, inputs, label

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, close_enough, driver  # noqa: E402

OP = "hstu_cross_attn"
REF_NAME = "tutorials/hstu_cross_attn/triton_bw_cross_attention.py:triton_bw_hstu_mha_wrapper"
#: No `heuristic_config` yet; see `bench_flash_attn.py`.
DEFAULT_SPACE = "full"
EXTRA_COLUMNS = (("q tokens", "q_tokens"), ("kv tokens", "kv_tokens"))
COLD_COMPILE = "first"
COMPILE_CAP_S = 900.0

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
DIRECTIONS = ("fwd", "bwd")
#: Inference captures: production never runs their backward, and both backends'
#: backward takes 60-530 ms a call on them, so the perf suite times the forward.
FORWARD_ONLY = frozenset(SHAPE_SUITES.resolved_shapes("gfx950_1"))

REL_PRECISION = {"float16": 1e-3, "bfloat16": 8e-3}

_TUTORIAL = pathlib.Path(__file__).resolve().parents[3] / "third_party" / "tlx" / "tutorials" / "hstu_cross_attn"


def _reference():
    if str(_TUTORIAL) not in sys.path:
        sys.path.insert(0, str(_TUTORIAL))
    return importlib.import_module("triton_bw_cross_attention").triton_bw_hstu_mha_wrapper


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(op=OP, arch=driver.arch(), dtype=str(DTYPES[entry.dtype]).removeprefix("torch."), shape=tuple(entry),
             direction=direction, label=label(entry, direction))
        for entry in shapes(synthetic, suites)
        for direction in (("fwd", ) if entry in FORWARD_ONLY else DIRECTIONS)
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops import hstu_cross_attn
    from triton.tlx.ops.kernels.hstu_cross_attn._shapes import HSTUCrossAttentionShape

    shape = HSTUCrossAttentionShape(*case.shape)
    backward = case.direction == "bwd"
    args = inputs(shape, requires_grad=backward)
    lengths_q, lengths_kv = args.pop("lengths_q"), args.pop("lengths_kv")
    ref_mha = _reference()

    tlx_fwd = lambda: hstu_cross_attn(space=space, **args)  # noqa: E731
    ref_fwd = lambda: ref_mha(  # noqa: E731
        max_seq_len=args["max_seq_len"], alpha=args["alpha"], q=args["q"], k=args["k"], v=args["v"], seq_offsets=args[
            "seq_offsets"], attn_scale=args["attn_scale"], max_q_len=args["max_q_len"], seq_offsets_q=args[
                "seq_offsets_q"], num_softmax_heads=args["num_softmax_heads"], num_targets=args["num_targets"],
        shared_kv=args["shared_kv"], enable_tma=False)
    extra = {"q_tokens": int(lengths_q.sum()), "kv_tokens": int(lengths_kv.sum()), "ref_autotuned": True}
    flop_count = flops(shape, lengths_q, lengths_kv, case.direction)

    if not backward:
        return Prepared(tlx_fn=tlx_fwd, ref_fn=ref_fwd, flop_count=flop_count,
                        check=lambda: close_enough(tlx_fwd(), ref_fwd(), REL_PRECISION[case.dtype]), extra=extra,
                        cap_s=COMPILE_CAP_S)

    tlx_out, ref_out = tlx_fwd(), ref_fwd()
    do = torch.randn_like(tlx_out)
    leaves = [args["q"], args["k"]] + ([] if args["shared_kv"] else [args["v"]])
    return Prepared(
        tlx_fn=lambda: tlx_out.backward(do, retain_graph=True),
        ref_fn=lambda: ref_out.backward(do, retain_graph=True),
        flop_count=flop_count,
        grad_to_none=leaves,
        check=None,  # see bench_flash_attn.py
        extra=extra,
        cap_s=COMPILE_CAP_S,
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
