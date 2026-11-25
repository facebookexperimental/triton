"""Single-site residual/SwiGLU backward and stochastic-store benchmarks."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import statistics

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

SR_SPEC = triton.StochasticRoundingSpec(
    version=1,
    site=triton.SRStoreSpec(
        name="output",
        source_dtype=tl.float32,
        destination_dtype=tl.bfloat16,
        site_id=0,
    ),
    prng="philox4x32-10",
)


@triton.jit
def _prototype_mix(value):
    value = value.to(tl.uint64)
    value ^= value >> 33
    value *= 0xff51afd7ed558ccd
    value ^= value >> 33
    value *= 0xc4ceb9fe1a85ec53
    return value ^ (value >> 33)


@triton.jit
def _prototype_round(value, index, seed, rounds: tl.constexpr):
    mixed = _prototype_mix(_prototype_mix(tl.load(seed)) ^ tl.full((), 23, tl.uint64))
    entropy = tl.randint(mixed, index, n_rounds=rounds).to(tl.uint32)
    rounded = tlx.stoch_round(value, tl.bfloat16, entropy ^ (entropy >> 16))
    special = (value.to(tl.uint32, bitcast=True) & 0x7f800000) == 0x7f800000
    return tl.where(special, value.to(tl.bfloat16), rounded)


@triton.jit
def _workload(A, B, G, Gate, Out, N: tl.constexpr, H: tl.constexpr, MODE: tl.constexpr, ANNOTATED: tl.constexpr,
              ManualSeed=None, MANUAL: tl.constexpr = False, ROUNDS: tl.constexpr = 10):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, H).to(tl.int64)
    index = row * H + col
    a = tl.load(A + index, row < N, 0).to(tl.float32)
    if MODE == 0:
        value = a
    else:
        b = tl.load(B + index, row < N, 0).to(tl.float32)
        g = tl.load(G + index, row < N, 0).to(tl.float32)
        if MODE == 1 or MODE == 3:
            gate = tl.load(Gate + col).to(tl.float32)
            beta = 1. / (1. + tl.exp(-gate))
            alpha_sq = (1. / (1. + tl.exp(gate))) * (1. + beta)
            alpha = tl.sqrt(tl.maximum(alpha_sq, 1.e-6))
            value = g * beta if MODE == 1 else g * alpha
        else:
            sigmoid = 1. / (1. + tl.exp(-a))
            if MODE == 2:
                value = g * b * sigmoid * (1. + a * (1. - sigmoid))
            else:
                value = g * a * sigmoid
    if MANUAL:
        tl.store(Out + index, _prototype_round(value, index, ManualSeed, ROUNDS), mask=row < N)
    elif ANNOTATED:
        tl.sr_store(Out + index, value, index, "output", mask=row < N)
    else:
        tl.store(Out + index, value, mask=row < N)


@triton.jit
def _gate_gradient(A, B, G, Gate, Out, N: tl.constexpr, H: tl.constexpr, T: tl.constexpr, C: tl.constexpr = 128,
                   ANNOTATED: tl.constexpr = True, ManualSeed=None, MANUAL: tl.constexpr = False,
                   ROUNDS: tl.constexpr = 10):
    hidden = tl.program_id(0).to(tl.int64) * C + tl.arange(0, C).to(tl.int64)
    token = tl.arange(0, T).to(tl.int64)
    d_alpha = tl.full((C, ), 0., tl.float32)
    d_beta = tl.full((C, ), 0., tl.float32)
    for block in range(tl.cdiv(N, T)):
        row = block.to(tl.int64) * T + token
        index = row[:, None] * H + hidden[None, :]
        mask = (row[:, None] < N) & (hidden[None, :] < H)
        a = tl.load(A + index, mask, 0)
        b = tl.load(B + index, mask, 0)
        g = tl.load(G + index, mask, 0)
        d_alpha += tl.sum(g * a, 0)
        d_beta += tl.sum(g * b, 0)
    gate = tl.load(Gate + hidden, hidden < H, 0)
    beta = 1. / (1. + tl.exp(-gate))
    minus = 1. / (1. + tl.exp(gate))
    alpha_sq = minus * (1. + beta)
    alpha = tl.sqrt(tl.maximum(alpha_sq, 1.e-6))
    d_alpha_sq = tl.where(alpha_sq > 1.e-6, d_alpha / (2. * alpha), 0.)
    value = (d_beta + d_alpha_sq * minus) * beta * (1. - beta)
    value -= d_alpha_sq * (1. + beta) * minus * (1. - minus)
    if MANUAL:
        tl.store(Out + hidden, _prototype_round(value, hidden, ManualSeed, ROUNDS), mask=hidden < H)
    elif ANNOTATED:
        tl.sr_store(Out + hidden, value, hidden, "output", mask=hidden < H)
    else:
        tl.store(Out + hidden, value, mask=hidden < H)


@triton.jit
def _cast(Input, Out, N: tl.constexpr, H: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * H + tl.arange(0, H).to(tl.int64)
    value = tl.load(Input + index, index < N, 0)
    tl.sr_store(Out + index, value, index, "output", mask=index < N)


def variant(kernel, rounds=10):
    spec = SR_SPEC if rounds == 10 else replace(SR_SPEC, prng=f"philox4x32-{rounds}")
    return triton.triton_stochastic_rounding(spec)(kernel)


def inputs(rows, width):
    torch.manual_seed(17)
    a, b, g = [torch.randn((rows, width), device="cuda", dtype=torch.float32) for _ in range(3)]
    gate = torch.randn(width, device="cuda", dtype=torch.float32)
    return a, b, g, gate


def reference(a, b, g, gate, mode):
    if mode == 0:
        return a
    if mode in (1, 3, 4):
        beta = torch.sigmoid(gate)
        alpha = (torch.sigmoid(-gate) * (1 + beta)).clamp_min(1.e-6).sqrt()
        if mode == 1:
            return g * beta
        if mode == 3:
            return g * alpha
        gate_leaf = gate.detach().clone().requires_grad_()
        beta = torch.sigmoid(gate_leaf)
        alpha = (torch.sigmoid(-gate_leaf) * (1 + beta)).clamp_min(1.e-6).sqrt()
        return torch.autograd.grad(alpha * a + beta * b, gate_leaf, g)[0]
    sigmoid = torch.sigmoid(a)
    return g * b * sigmoid * (1 + a * (1 - sigmoid)) if mode == 2 else g * a * sigmoid


def _measure(fn):
    samples = triton.testing.do_bench_cudagraph(fn, rep=100, return_mode="all")
    return dict(median_us=statistics.median(samples) * 1000, min_us=min(samples) * 1000, max_us=max(samples) * 1000)


def _capture(directory, label, kernel, args, grid, expect_sr=False, **kwargs):
    compiled = kernel.warmup(*args, grid=grid, **kwargs)
    compiled._init_handles()
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for stage, extension in (("ttir", "ttir"), ("ttgir", "ttgir"), ("llir", "ll"), ("ptx", "ptx"), ("sass", "sass"),
                             ("cubin", "cubin")):
        data = compiled.asm[stage]
        if isinstance(data, str):
            data = data.encode()
        if not data:
            raise RuntimeError(f"Missing {stage} for {directory.name}/{label}")
        path = directory / f"{label}.{extension}"
        path.write_bytes(data)
        files[stage] = dict(path=path.name, sha256=hashlib.sha256(data).hexdigest())
    ptx_rs = re.findall(r"\bcvt\.rs(?:\.satfinite)?\.bf16x2\.f32\b", compiled.asm["ptx"])
    sass_rs = [
        line.strip()
        for line in compiled.asm["sass"].splitlines()
        if re.search(r"\bF2FP(?:\.SATFINITE)?\.BF16\.F32\.PACK_AB\.RS\b", line)
    ]
    record = dict(case=directory.name, variant=label, kernel=compiled.metadata.name, kernel_hash=compiled.hash,
                  grid=grid, num_warps=compiled.metadata.num_warps, registers=compiled.n_regs, spills=compiled.n_spills,
                  files=files, expected_sr=expect_sr, ptx_sr_count=len(ptx_rs), sass_sr_count=len(sass_rs),
                  ptx_sr_opcodes=sorted(set(ptx_rs)), sass_sr_examples=sass_rs[:2])
    (directory / f"{label}.json").write_text(json.dumps(record, indent=2) + "\n")
    if bool(ptx_rs) != expect_sr or bool(sass_rs) != expect_sr:
        raise RuntimeError(f"Unexpected stochastic conversion instructions for {directory.name}/{label}")
    if label in ("fused", "unfused_cast") and set(ptx_rs) != {"cvt.rs.bf16x2.f32"}:
        raise RuntimeError(f"Expected IEEE cvt.rs.bf16x2.f32 for {directory.name}/{label}")
    return record


def benchmark(rows, width, rounds, dump_dir=None, capture_only=False):
    a, b, g, gate = inputs(rows, width)
    seed = torch.tensor([19], device="cuda", dtype=torch.int64)
    cast = variant(_cast, rounds)
    results = []
    captures = []
    cases = ((0, "identity"), (1, "residual_grad_branch"), (3, "residual_grad_stream"), (4, "residual_grad_gate"),
             (2, "swiglu_grad_a"), (5, "swiglu_grad_b"))
    for mode, name in cases:
        source = _gate_gradient if mode == 4 else _workload
        fused = variant(source, rounds)
        out = torch.empty_like(gate if mode == 4 else a, dtype=torch.bfloat16)
        scratch = torch.empty_like(out, dtype=torch.float32)
        grid = (triton.cdiv(width, 128), ) if mode == 4 else (rows, )
        tail = (rows, width, 32) if mode == 4 else (rows, width, mode)
        args = (a, b, g, gate, out, *tail)
        cast_block = 128 if mode == 4 else width
        cast_grid = (triton.cdiv(out.numel(), cast_block), )
        directory = Path(dump_dir) / name if dump_dir else None
        if directory is not None:
            captures.append(_capture(directory, "plain", fused, args, grid, ANNOTATED=True))
            captures.append(_capture(directory, "original_plain", source, args, grid, ANNOTATED=False))
            captures.append(
                _capture(directory, "unfused_math", source, (a, b, g, gate, scratch, *tail), grid, ANNOTATED=False))
        plain = lambda: fused[grid](*args, ANNOTATED=True)
        with triton.stochastic_rounding(dynamic_seed=seed, base_seed=23):
            if directory is not None:
                captures.append(_capture(directory, "fused", fused, args, grid, True, ANNOTATED=True))
                captures.append(
                    _capture(directory, "unfused_cast", cast, (scratch, out, out.numel(), cast_block), cast_grid, True))
            sr = lambda: fused[grid](*args, ANNOTATED=True)

            def unfused():
                source[grid](a, b, g, gate, scratch, *tail, ANNOTATED=False)
                cast[cast_grid](scratch, out, out.numel(), cast_block)

            sr()
            expected = out.clone()
            unfused()
            torch.testing.assert_close(out.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
            if not capture_only:
                fused_time = _measure(sr)
                unfused_time = _measure(unfused)
        if directory is not None:
            captures.append(
                _capture(directory, "prototype", source, args, grid, True, ANNOTATED=False, ManualSeed=seed,
                         MANUAL=True, ROUNDS=rounds))
        prototype = lambda: source[grid](*args, ANNOTATED=False, ManualSeed=seed, MANUAL=True, ROUNDS=rounds)
        plain()
        torch.testing.assert_close(out.float(), reference(a, b, g, gate, mode), rtol=.02, atol=.03)
        prototype()
        prototype_error = (out.float() - reference(a, b, g, gate, mode)).abs().max().item()
        if capture_only:
            continue
        prototype_time = _measure(prototype)
        plain_time = _measure(plain)
        original_plain_time = _measure(lambda: source[grid](*args, ANNOTATED=False))
        results.append(
            dict(
                case=name, rows=rows, width=width, rounds=rounds, plain_us=plain_time["median_us"],
                fused_sr_us=fused_time["median_us"], original_plain_us=original_plain_time["median_us"],
                unfused_sr_us=unfused_time["median_us"], prototype_sr_us=prototype_time["median_us"],
                speedup=unfused_time["median_us"] / fused_time["median_us"], prototype_max_abs_error=prototype_error,
                samples=dict(plain=plain_time, original_plain=original_plain_time, fused=fused_time,
                             unfused=unfused_time, prototype=prototype_time)))
    if dump_dir:
        manifest = dict(device=torch.cuda.get_device_name(), compute_capability=torch.cuda.get_device_capability(),
                        rows=rows, width=width, rounds=rounds, triton_version=triton.__version__,
                        torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        cuobjdump=vars(triton.knobs.nvidia.cuobjdump),
                        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), captures=captures)
        (Path(dump_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--rounds", type=int, choices=(7, 10), default=10)
    parser.add_argument("--dump-dir", type=Path, help="Save TTIR/TTGIR/LLVM IR/PTX/SASS and cubins")
    parser.add_argument("--capture-only", action="store_true", help="Check outputs and capture code without timing")
    args = parser.parse_args()
    if args.rows <= 0 or args.width <= 0 or args.width & (args.width - 1):
        parser.error("rows must be positive and width must be a positive power of two")
    if args.capture_only and args.dump_dir is None:
        parser.error("--capture-only requires --dump-dir")
    print(
        json.dumps(
            dict(device=torch.cuda.get_device_name(),
                 timing=None if args.capture_only else "CUDA graph median of 10 replays",
                 repetition_budget_ms=None if args.capture_only else 100, buffer_rotation=False, cache_flush=False,
                 capture_dir=str(args.dump_dir.resolve()) if args.dump_dir else None,
                 results=benchmark(args.rows, args.width, args.rounds, args.dump_dir, args.capture_only)), indent=2))


if __name__ == "__main__":
    main()
