"""Compile the paper's sub-tiled FA backward fixture on Blackwell.

The default invocation uses the paper shape and enables the modulo-schedule
source marker.  Set MLIR_ENABLE_DUMP and MLIR_DUMP_PATH as documented in the
adjacent README to capture the pre-modulo TTGIR.  ``--check`` instead launches
the no-WS form and compares a smaller shape with torch autograd.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import triton

from .fa_bwd_dkdv_subtiled import fa_bwd_dkdv_subtiled

LOG2E = 1.4426950408889634


def _allocator(size: int, alignment: int, stream: int) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q, k.transpose(-1, -2))
    output = torch.matmul(torch.softmax(scores, dim=-1), v)
    normalizer = torch.logsumexp(scores, dim=-1) * LOG2E
    return output, normalizer


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = (actual.float() - expected.float()).abs().max().item()
    denominator = max(expected.float().abs().max().item(), 1e-9)
    return numerator / denominator


def _requested_dump_paths() -> tuple[Path, ...]:
    dump_variables = (
        "TRITON_MODULO_DUMP_DDG",
        "TRITON_MODULO_DUMP_SCHEDULE",
    )
    return tuple(
        Path(value)
        for variable in dump_variables
        if (value := os.environ.get(variable))
    )


def _check_requested_dumps(paths: tuple[Path, ...]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"requested compiler dumps were not written: {missing}")


def run(
    *,
    bh: int,
    n_ctx: int,
    head_dim: int,
    sub_m: int,
    block_n: int,
    check: bool,
) -> bool:
    triton.set_allocator(_allocator)
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(head_dim)
    q = (
        torch.randn(bh, n_ctx, head_dim, device="cuda", dtype=torch.float16)
        * scale
    ).requires_grad_(check)
    k = torch.randn(
        bh, n_ctx, head_dim, device="cuda", dtype=torch.float16
    ).requires_grad_(check)
    v = torch.randn(
        bh, n_ctx, head_dim, device="cuda", dtype=torch.float16
    ).requires_grad_(check)
    do = torch.randn(bh, n_ctx, head_dim, device="cuda", dtype=torch.float16)

    if check:
        output, normalizer = _reference(q, k, v)
        output.backward(do)
        expected = (q.grad, k.grad, v.grad)
        D = (do.float() * output.detach().float()).sum(-1).contiguous()
    else:
        normalizer = torch.zeros(
            bh, n_ctx, device="cuda", dtype=torch.float32
        )
        D = torch.zeros_like(normalizer)
        expected = (None, None, None)

    dq = torch.zeros_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    grid = (n_ctx // block_n, bh)
    fa_bwd_dkdv_subtiled[grid](
        q.detach(),
        k.detach(),
        v.detach(),
        do,
        dq,
        dk,
        dv,
        normalizer.detach().float().contiguous(),
        D,
        head_dim,
        head_dim,
        n_ctx,
        SUB_M=sub_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        WARP_SPEC=not check,
        num_warps=4,
        num_ctas=1,
        num_stages=1,
    )
    torch.cuda.synchronize()

    if not check:
        return True
    errors = tuple(
        _relative_error(actual, reference)
        for actual, reference in zip((dq, dk, dv), expected)
        if reference is not None
    )
    print(
        f"dQ={errors[0]:.2e} dK={errors[1]:.2e} dV={errors[2]:.2e}"
    )
    return max(errors) < 3e-2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return 0 if run(
            bh=1,
            n_ctx=512,
            head_dim=64,
            sub_m=64,
            block_n=128,
            check=True,
        ) else 1
    dump_paths = _requested_dump_paths()
    for path in dump_paths:
        path.unlink(missing_ok=True)
    try:
        run(
            bh=4,
            n_ctx=2048,
            head_dim=128,
            sub_m=64,
            block_n=128,
            check=False,
        )
    except RuntimeError as error:
        # The modulo pass can intentionally stop compilation after emitting its
        # dumps; reaching it is sufficient for this fixture runner.
        if not dump_paths:
            raise
        _check_requested_dumps(dump_paths)
        first_line = str(error).splitlines()[0]
        print(f"launch stopped after compilation: {type(error).__name__}: {first_line}")
    _check_requested_dumps(dump_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
