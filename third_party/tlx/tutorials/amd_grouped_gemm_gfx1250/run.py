"""Run one gfx1250 grouped GEMM case.

The shape-sweep driver invokes this module in a fresh process for each case so
large device allocations are released between measurements.
"""

from __future__ import annotations

import argparse
from itertools import accumulate

import torch

import triton
from triton._internal_testing import is_hip_gfx1250

if __package__:
    from .grouped_gemm import _active_num_cus, _pick_grouped_gemm_config, grouped_gemm_tdm
else:
    from grouped_gemm import _active_num_cus, _pick_grouped_gemm_config, grouped_gemm_tdm


def _parse_m_list(value: str) -> list[int]:
    result = [int(field) for field in value.split(",") if field]
    if not result:
        raise argparse.ArgumentTypeError("m-list must contain at least one group")
    if min(result) <= 0:
        raise argparse.ArgumentTypeError("all group M sizes must be positive")
    return result


def _make_inputs(m_list: list[int], n: int, k: int, device: torch.device, keep_groups: bool):
    if keep_groups:
        group_a = [torch.randn((m, k), device=device, dtype=torch.float16) for m in m_list]
        a_packed = torch.cat(group_a, dim=0).contiguous()
    else:
        group_a = None
        a_packed = torch.randn((sum(m_list), k), device=device, dtype=torch.float16)
    b_t = torch.randn((len(m_list), n, k), device=device, dtype=torch.float16)
    offsets = [0, *accumulate(m_list)]
    group_offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    return a_packed, b_t, group_offsets, group_a


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one TLX AMD f16 grouped GEMM TDM case")
    parser.add_argument("--m-list", "--m_list", dest="m_list", type=_parse_m_list,
                        default=_parse_m_list("512,512,512,512"), help="comma-separated M sizes per group")
    parser.add_argument("-N", type=int, default=1024, help="problem N size")
    parser.add_argument("-K", type=int, default=2048, help="problem K size")
    parser.add_argument("-BM", type=int, default=256, help="BLOCK_M")
    parser.add_argument("-BN", type=int, default=256, help="BLOCK_N")
    parser.add_argument("-BK", type=int, default=128, help="BLOCK_K")
    parser.add_argument("--group-m", "--group_m", dest="group_m", type=int, default=4,
                        help="GROUP_M launch-order swizzle")
    parser.add_argument("--tdm-pipeline-depth", "--tdm_pipeline_depth", dest="tdm_pipeline_depth", type=int, default=2,
                        choices=(2, 3, 4), help="TDM LDS ring-buffer depth")
    parser.add_argument("--l2-prefetch-distance", "--l2_prefetch_distance", dest="l2_prefetch_distance", type=int,
                        default=0)
    parser.add_argument("--num-programs", "--num_programs", dest="num_programs", type=int, default=None,
                        help="persistent program count; defaults to the active CU count")
    parser.add_argument("--dedicated-c-buffer", "--dedicated_c_buffer", dest="dedicated_c_buffer",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="stage C in distinct LDS instead of aliasing the A ring")
    parser.add_argument("--cross-tile-prefetch", "--cross_tile_prefetch", dest="cross_tile_prefetch",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="prefetch the next same-group tile in the peeled K-loop tail")
    parser.add_argument("--auto-config", "--auto_config", dest="auto_config", action=argparse.BooleanOptionalAction,
                        default=False, help="select a validated tile config using the host cost model")
    parser.add_argument("--xcd-remap", "--xcd_remap", dest="xcd_remap", choices=("none", "balanced", "chunked"),
                        default="none", help="persistent program-id mapping across gfx1250 XCDs")
    parser.add_argument("--num-xcds", "--num_xcds", dest="num_xcds", type=int, default=8)
    parser.add_argument("--xcd-chunk", "--xcd_chunk", dest="xcd_chunk", type=int, default=2)
    parser.add_argument("--benchmark-mode", "--benchmark_mode", dest="benchmark_mode",
                        choices=("eager", "graph", "none"), default="eager")
    parser.add_argument("--benchmark-num-iters", "--benchmark_num_iters", dest="benchmark_num_iters", type=int,
                        default=32)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _create_parser().parse_args(argv)
    if not is_hip_gfx1250():
        raise SystemExit("Requires gfx1250 hardware")

    device = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(args.seed)
    a_packed, b_t, group_offsets, group_a = _make_inputs(args.m_list, args.N, args.K, device, args.check)

    if args.auto_config:
        model_sms = args.num_programs if args.num_programs is not None else _active_num_cus(device)
        selected = _pick_grouped_gemm_config(args.m_list, args.N, args.K, model_sms)
        print("selected config:", selected["name"], f"BM={selected['block_m']}", f"BN={selected['block_n']}",
              f"cross_tile_prefetch={selected['cross_tile_prefetch']}")

    benchmark = None if args.benchmark_mode == "none" else args.benchmark_mode
    out = grouped_gemm_tdm(
        a_packed,
        b_t,
        group_offsets,
        block_m=args.BM,
        block_n=args.BN,
        block_k=args.BK,
        group_m=args.group_m,
        tdm_pipeline_depth=args.tdm_pipeline_depth,
        l2_prefetch_distance=args.l2_prefetch_distance,
        num_programs=args.num_programs,
        c_staging_mode=int(args.dedicated_c_buffer),
        cross_tile_prefetch=args.cross_tile_prefetch,
        auto_config=args.auto_config,
        xcd_remap_mode=args.xcd_remap,
        num_xcds=args.num_xcds,
        xcd_chunk=args.xcd_chunk,
        benchmark=benchmark,
        benchmark_num_iters=args.benchmark_num_iters,
    )

    if args.check:
        assert group_a is not None
        start = 0
        max_diff = 0.0
        for index, m in enumerate(args.m_list):
            ref = group_a[index] @ b_t[index].T
            actual = out[start:start + m]
            max_diff = max(max_diff, (actual - ref).abs().max().item())
            torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)
            start += m
        print(f"max abs diff: {max_diff:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
