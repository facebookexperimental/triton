"""L2 performance coverage for ``tlx.ops.flash_attn_varlen_mxfp8``.

K/V preparation and the dispatch tables are reusable state and stay outside
the measured closure: every measured call passes the same offset tensors, so
it runs the attention kernel alone. Correctness is covered by the op's L1
tests; this benchmark has no reference provider.
"""

from __future__ import annotations

import pathlib
import sys

import torch
from triton.tlx.ops.kernels.flash_attn_varlen_mxfp8._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.flash_attn_varlen_mxfp8._shapes import (SYNTHETIC, FlashAttentionVarlenMXFP8Shape, flops,
                                                                    inputs, label, lengths)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver  # noqa: E402

OP = "flash_attn_varlen_mxfp8"
REF_NAME = ""
DEFAULT_SPACE = "heuristic"
EXTRA_COLUMNS = (("q_tokens", "q_tokens"), ("kv_tokens", "kv_tokens"))
COLD_COMPILE = "first"

DTYPES = {"bf16": torch.bfloat16}


def shapes(synthetic: bool = False, suites=None) -> list:
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    return [
        Case(
            op=OP,
            arch=driver.arch(),
            dtype=str(DTYPES[entry.dtype]).removeprefix("torch."),
            shape=tuple(entry[:-1]),
            label=label(*entry),
        ) for entry in shapes(synthetic, suites)
    ]


def prepare(case: Case, space: str) -> Prepared:
    from triton.tlx.ops.kernels.flash_attn_varlen_mxfp8 import gfx950

    del space  # One fixed launch config.
    shape = FlashAttentionVarlenMXFP8Shape(*case.shape, "bf16")
    query_lengths, key_lengths = lengths(shape)
    q, k, v, cu_q, cu_k = inputs(query_lengths, key_lengths, getattr(torch, case.dtype))
    metadata = gfx950._metadata(cu_q, cu_k, q.shape[0], k.shape[0], 128)
    kv_state = gfx950.prepare_kv(k, v, cu_k, metadata)

    def tlx_fwd():
        return gfx950.forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            128**-0.5,
            kv_state=kv_state,
        )[0]

    return Prepared(
        tlx_fn=tlx_fwd,
        ref_fn=None,
        flop_count=flops(query_lengths, key_lengths),
        check=None,
        extra={"q_tokens": q.shape[0], "kv_tokens": k.shape[0]},
    )


supported, default_json, run, main = driver.bind(sys.modules[__name__])

if __name__ == "__main__":
    raise SystemExit(main())
