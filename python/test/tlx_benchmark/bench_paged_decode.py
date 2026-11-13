"""Public BF16 paged decode: reusable buffers, full FP64 checks and GPU latency."""

import pathlib
import sys

import torch
from triton.tlx.ops import paged_attention_decode, paged_attention_decode_workspace
from triton.tlx.ops.kernels.paged_attn._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.paged_attn._shapes import SYNTHETIC, flops
from triton.tlx.ops.kernels.paged_attn._testing import assert_close, make_inputs, pack_cache, reference

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import Case, Prepared, driver

OP = "paged_attention_decode"
REF_NAME = ""
DEFAULT_SPACE = "heuristic"
COLD_COMPILE = "first"
EXTRA_COLUMNS = (("latency us", "latency_us"), )


def shapes(synthetic=False, suites=None):
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic=False, suites=None):
    return [
        Case(op=OP, arch=driver.arch(), dtype="bfloat16", shape=s,
             label=f"B={s[0]} Q={s[1]} KV={s[2]} window={s[3]} cache={'5d' if s[4] else '4d'}")
        for s in shapes(synthetic, suites)
    ]


def prepare(case, space):
    batch, query_length, context, window, packed = case.shape
    inputs = make_inputs([context] * batch, query_length)
    q, k, v, lengths, table = inputs
    if packed:
        k, v = pack_cache(k, v)
    workspace = paged_attention_decode_workspace(q, k, v, table, query_length=query_length, sliding_window=window,
                                                 max_context_len=context)
    out = torch.empty_like(q)
    fn = lambda: paged_attention_decode(q, k, v, lengths, table, query_length=query_length, sliding_window=window,
                                        max_context_len=context, out=out, workspace=workspace, space=space)

    def check():
        try:
            rrms = assert_close(fn(), reference(*inputs, query_length, window))
            return True, f"FP64 relative RMS {rrms:.5f}"
        except AssertionError as error:
            return False, str(error)

    return Prepared(tlx_fn=fn, ref_fn=None, flop_count=flops(*case.shape), check=check)


def annotate(result):
    if result.tlx and result.flop_count:
        result.extra["latency_us"] = result.flop_count / (result.tlx.mean * 1e6)


_supported, default_json, run, main = driver.bind(sys.modules[__name__])


def supported():
    return driver.arch() == "gfx1250" and _supported()


if __name__ == "__main__":
    raise SystemExit(main())
