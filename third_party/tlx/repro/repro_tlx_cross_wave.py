"""gfx950 cross-wave MFMA32 / packed-FP32 subtraction reproducer.

TLX selects four MFMA waves and four ADD waves in one eight-wave workgroup.
Inline assembly fixes the exact opcodes under test; no attention, LDS, or
assembly override is involved. Exit 2 means a numerical mismatch was found.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _mfma_body(sink, lane, ASM: tl.constexpr, CONS: tl.constexpr):
    sink, counter = tl.inline_asm_elementwise(
        ASM,
        constraints=CONS,
        args=[lane],
        dtype=(tl.float32, tl.int32),
        is_pure=False,
        pack=1,
    )
    return sink


@triton.jit
def _add_body(bad, lo, hi, first_iter, lane, ASM: tl.constexpr, CONS: tl.constexpr):
    return tl.inline_asm_elementwise(
        ASM,
        constraints=CONS,
        args=[lane],
        dtype=(tl.int32, tl.float32, tl.float32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )[:4]


@triton.jit
def repro(
    Bad,
    Lo,
    Hi,
    Iter,
    Sink,
    MFMA_ASM: tl.constexpr,
    MFMA_CONS: tl.constexpr,
    ADD_ASM: tl.constexpr,
    ADD_CONS: tl.constexpr,
):
    x = tl.arange(0, 512)
    lane = x % 64
    # Physical thread ID makes the predicate uniform over every complete wave.
    mfma_wave = tlx.thread_id(0) < 256
    sink = tl.zeros((512,), tl.float32)
    sink = tlx.warp_predicate(
        mfma_wave, sink, _mfma_body, args=(lane, MFMA_ASM, MFMA_CONS)
    )
    zero_i = tl.zeros((512,), tl.int32)
    zero_f = tl.zeros((512,), tl.float32)
    bad, lo, hi, first_iter = tlx.warp_predicate(
        ~mfma_wave,
        (zero_i, zero_f, zero_f, zero_i - 1),
        _add_body,
        args=(lane, ADD_ASM, ADD_CONS),
    )
    offsets = tl.program_id(0) * 512 + x
    tl.store(Bad + offsets, bad)
    tl.store(Lo + offsets, lo)
    tl.store(Hi + offsets, hi)
    tl.store(Iter + offsets, first_iter)
    tl.store(Sink + offsets, sink)


def mfma_source(mode: int, iterations: int) -> tuple[str, str]:
    lines = [f"v_mov_b32 v{i}, 0" for i in range(32, 48)]
    lines += [
        "v_mov_b32 v48, $2",
        "v_add_u32 v49, 1, v48",
        "v_add_u32 v50, 2, v48",
        "v_add_u32 v51, 3, v48",
    ]
    lines += [f"v_mov_b32 v{i}, {i - 51}" for i in range(52, 56)]
    lines += [f"s_mov_b32 $1, {iterations}", "1:"]
    if mode in (0, 1):
        k = 32 if mode == 0 else 16
        a, b = ("v[48:51]", "v[52:55]") if k == 32 else ("v[48:49]", "v[52:53]")
        lines += [
            f"v_mfma_f32_16x16x{k}_f16 v[{i}:{i + 3}], {a}, {b}, v[{i}:{i + 3}]"
            for i in (32, 36, 40, 44)
        ]
    elif mode == 2:
        lines += [f"v_mov_b32 v{i}, $1" for i in (32, 36, 40, 44)]
    else:
        lines += ["s_nop 0"]
    lines += [
        "s_sub_u32 $1, $1, 1",
        "s_cmp_lg_u32 $1, 0",
        "s_cbranch_scc1 1b",
        "s_nop 15",
        "s_nop 15",
        "v_add_f32 v32, v32, v37",
        "v_add_f32 v32, v32, v42",
        "v_add_f32 $0, v32, v47",
    ]
    cons = "=v,=&s,v," + ",".join(f"~{{v{i}}}" for i in range(32, 56)) + ",~{scc}"
    return "\n".join(lines), cons


def add_source(form: int, iterations: int) -> tuple[str, str]:
    lines = [
        "v_cvt_f32_u32 v40, $5",
        "v_add_f32 v32, 0x42dc0000, v40",
        "v_add_f32 v33, 0x437d0000, v40",
        "v_mov_b32 v34, 0",
        "v_mul_f32 v35, 0.25, v40",
        "v_add_f32 v35, 0x44200000, v35",
        "v_sub_f32 v38, v32, v35",
        "v_sub_f32 v39, v33, v35",
    ]
    if form == 2:
        lines += ["v_mov_b32 v34, v35"]
    lines += [
        "v_mov_b32 $0, 0",
        "v_mov_b32 $1, 0",
        "v_mov_b32 $2, 0",
        "v_mov_b32 $3, -1",
        f"s_mov_b32 $4, {iterations}",
        "2:",
    ]
    forms = [
        "v_pk_add_f32 v[36:37], v[32:33], v[34:35] op_sel:[0,1] neg_lo:[0,1] neg_hi:[0,1]",
        "v_pk_add_f32 v[36:37], v[34:35], v[32:33] op_sel:[1,0] neg_lo:[1,0] neg_hi:[1,0]",
        "v_pk_add_f32 v[36:37], v[32:33], v[34:35] neg_lo:[0,1] neg_hi:[0,1]",
    ]
    lines += [
        forms[form],
        "v_cmp_neq_f32_e32 vcc, v36, v38",
        "v_cndmask_b32_e64 v41, 0, 1, vcc",
        "v_cmp_neq_f32_e32 vcc, v37, v39",
        "v_cndmask_b32_e64 v42, 0, 1, vcc",
        "v_or_b32 v41, v41, v42",
        "v_cmp_eq_u32_e32 vcc, 0, $0",
        "v_cndmask_b32_e64 v43, 0, 1, vcc",
        "v_and_b32 v43, v43, v41",
        "v_mov_b32 v42, $4",
        f"v_sub_u32 v42, {iterations}, v42",
        "v_cmp_ne_u32_e32 vcc, 0, v43",
        "v_cndmask_b32_e32 $1, $1, v36, vcc",
        "v_cndmask_b32_e32 $2, $2, v37, vcc",
        "v_cndmask_b32_e32 $3, $3, v42, vcc",
        "v_or_b32 $0, $0, v41",
        "s_sub_u32 $4, $4, 1",
        "s_cmp_lg_u32 $4, 0",
        "s_cbranch_scc1 2b",
    ]
    cons = (
        "=v,=v,=v,=v,=&s,v,"
        + ",".join(f"~{{v{i}}}" for i in range(32, 44))
        + ",~{vcc},~{scc}"
    )
    return "\n".join(lines), cons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=int,
        choices=range(4),
        default=0,
        help="0: MFMA32, 1: MFMA16, 2: VALU, 3: NOP",
    )
    parser.add_argument(
        "--form",
        type=int,
        choices=range(3),
        default=0,
        help="0: failing src1-high, 1: commuted, 2: no op_sel",
    )
    parser.add_argument(
        "--waves-per-eu",
        type=int,
        choices=range(9),
        default=0,
        help="0: no occupancy cap; 1-8: compiler waves-per-SIMD setting",
    )
    parser.add_argument("--launches", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=1024)
    parser.add_argument("--mfma-iters", type=int, default=4096)
    parser.add_argument("--add-iters", type=int, default=65536)
    args = parser.parse_args()
    if min(args.launches, args.blocks, args.mfma_iters, args.add_iters) < 1:
        parser.error("launch counts, grid size, and iteration counts must be positive")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx950":
        parser.error(f"requires gfx950; found {target}")
    n = args.blocks * 512
    bad, first_iter = [
        torch.empty(n, device="cuda", dtype=torch.int32) for _ in range(2)
    ]
    lo, hi, sink = [
        torch.empty(n, device="cuda", dtype=torch.float32) for _ in range(3)
    ]
    mfma_asm, mfma_cons = mfma_source(args.mode, args.mfma_iters)
    add_asm, add_cons = add_source(args.form, args.add_iters)
    bad_launches, lane_mask, first = 0, 0, None
    for launch in range(args.launches):
        kernel = repro[(args.blocks,)](
            bad,
            lo,
            hi,
            first_iter,
            sink,
            mfma_asm,
            mfma_cons,
            add_asm,
            add_cons,
            num_warps=8,
            num_stages=1,
            waves_per_eu=args.waves_per_eu,
        )
        torch.cuda.synchronize()
        indices = torch.nonzero(bad).flatten().cpu().tolist()
        if indices:
            bad_launches += 1
            for index in indices:
                lane_mask |= 1 << (index % 64)
            if first is None:
                index = indices[0]
                lane = index % 64
                first = {
                    "block": index // 512,
                    "wave": index % 512 // 64,
                    "lane": lane,
                    "iter": int(first_iter[index]),
                    "lo": float(lo[index]),
                    "hi": float(hi[index]),
                    "want_lo": 110 + lane - (640 + 0.25 * lane),
                    "want_hi": 253 + lane - (640 + 0.25 * lane),
                }
        if launch == 0:
            output = Path(__file__).with_name(
                f"tlx_cross_wave_m{args.mode}_f{args.form}_w{args.waves_per_eu}.amdgcn"
            )
            output.write_text(kernel.asm["amdgcn"])
            print(
                json.dumps(
                    {
                        "asm": str(output),
                        "vgpr": kernel.n_regs,
                        "spills": kernel.n_spills,
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                **vars(args),
                "bad_launches": bad_launches,
                "lane_mask": f"0x{lane_mask:016x}",
                "first": first,
            }
        ),
        flush=True,
    )
    return 2 if bad_launches else 0


if __name__ == "__main__":
    raise SystemExit(main())
