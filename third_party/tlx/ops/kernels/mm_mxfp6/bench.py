import argparse

import torch
import triton
from triton._internal_testing import is_blackwell
from triton.tlx.ops import mm_mxfp6, mm_mxfp8
from triton.tlx.ops.kernels.mm._shapes import FOCUS, SYNTHETIC

# Benchmarks for the TLX MXFP6 GEMM op (A=E2M3, B=E3M2) on Blackwell.
# Run with (from the fbsource root, on a B200/GB200 box):
#   buck2 run @fbcode//mode/opt -m ovr_config//triton:beta \
#       -c fbcode.nvcc_arch=b200a -c fbcode.enable_gpu_sections=true \
#       fbsource//third-party/triton/beta/triton:py_tlx_mm_mxfp6_perf -- \
#       --shapes small --compare-mxfp8
#
# Prefix with third_party/tlx/denoise.sh for clock-locked numbers.


def _e8m0(shape, *, device, seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(124, 129, shape, device=device, dtype=torch.uint8,
                         generator=generator).view(torch.float8_e8m0fnu)


def _pack_fp6_dense(codes):
    # codes: uint8 [..., K] with values 0..63, K % 4 == 0. Packs 4 codes per
    # 3 bytes LSB-first and pads the K dim to the unpacked width with zeros.
    assert codes.shape[-1] % 4 == 0
    quads = codes.reshape(*codes.shape[:-1], -1, 4).to(torch.int64)
    v0, v1, v2, v3 = (quads[..., i] for i in range(4))
    b0 = v0 | ((v1 & 3) << 6)
    b1 = ((v1 >> 2) & 15) | ((v2 & 15) << 4)
    b2 = ((v2 >> 4) & 3) | (v3 << 2)
    triples = torch.stack([b0, b1, b2], dim=-1).reshape(*codes.shape[:-1], -1)
    padded = torch.zeros(codes.shape, dtype=torch.uint8, device=codes.device)
    padded[..., :triples.shape[-1]] = triples.to(torch.uint8)
    return padded


def _unpack_fp6_dense(packed):
    # Inverse of _pack_fp6_dense; ignores the trailing padding quarter.
    assert packed.shape[-1] % 4 == 0
    triples = packed[..., :3 * packed.shape[-1] // 4].to(torch.int64)
    triples = triples.reshape(*packed.shape[:-1], -1, 3)
    b0, b1, b2 = (triples[..., i] for i in range(3))
    v0 = b0 & 63
    v1 = ((b0 >> 6) & 3) | ((b1 & 15) << 2)
    v2 = ((b1 >> 4) & 15) | ((b2 & 3) << 4)
    v3 = (b2 >> 2) & 63
    return torch.stack([v0, v1, v2, v3], dim=-1).reshape(packed.shape).to(torch.uint8)


def _decode_fp6(codes, ebits, bias):
    mbits = 5 - ebits
    s = (codes >> 5) & 1
    e = (codes >> mbits) & ((1 << ebits) - 1)
    m = codes & ((1 << mbits) - 1)
    normal = (1.0 + m.float() / 2**mbits) * torch.pow(2.0, e.float() - bias)
    denormal = (m.float() / 2**mbits) * 2.0**(1 - bias)
    mag = torch.where(e == 0, denormal, normal)
    return torch.where(s == 1, -mag, mag)


def _dequantize(data, scale, *, ebits, bias):
    decoded = torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127)
    return _decode_fp6(_unpack_fp6_dense(data), ebits, bias) * decoded.repeat_interleave(32, dim=-1)


def _to_mxfp8(a, b):
    a = _decode_fp6(_unpack_fp6_dense(a), ebits=2, bias=1).to(torch.float8_e4m3fn)
    b = _decode_fp6(_unpack_fp6_dense(b), ebits=3, bias=3).to(torch.float8_e4m3fn)
    return a, b


def _make_inputs(m, n, k, device):
    generator = torch.Generator(device=device).manual_seed(0)
    a_codes = torch.randint(0, 64, (m, k), device=device, generator=generator, dtype=torch.uint8)
    b_codes = torch.randint(0, 64, (n, k), device=device, generator=generator, dtype=torch.uint8)
    a = _pack_fp6_dense(a_codes)
    b = _pack_fp6_dense(b_codes)
    a_scale = _e8m0((m, k // 32), device=device, seed=1)
    b_scale = _e8m0((n, k // 32), device=device, seed=2)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    return a, a_scale, b, b_scale, out


def _check_mxfp8_ones(m, n, k, device):
    # Sanity: the known-good mxfp8 op with fp8 ones (e4m3 code 0x38) and unit
    # scales must produce K. Validates the harness, not the fp6 kernel.
    a = torch.full((m, k), 0x38, device=device,
                 dtype=torch.uint8).view(torch.float8_e4m3fn)
    b = torch.full((n, k), 0x38, device=device,
                 dtype=torch.uint8).view(torch.float8_e4m3fn)
    a_scale = torch.full((m, k // 32), 127, device=device,
                         dtype=torch.uint8).view(torch.float8_e8m0fnu)
    b_scale = torch.full((n, k // 32), 127, device=device,
                         dtype=torch.uint8).view(torch.float8_e8m0fnu)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    actual = mm_mxfp8(a, a_scale, b, b_scale, out=out, space="heuristic")
    print(f"mxfp8-ones[{m},{n},{k}]: min={actual.float().min().item()} "
          f"max={actual.float().max().item()} mean={actual.float().mean().item()} (expect {k})")
    torch.testing.assert_close(actual.float(),
                               torch.full((m, n), float(k), device=device), atol=1e-1,
                               rtol=1e-2)


def _check_probe(m, n, k, device, space, kpos, *, side="both", a_code=8, b_code=12):
    # Single-element probe: unit scales; the probed side has a single code at
    # (0,kpos) and 0 elsewhere, the other side is all ones. Isolates A-side
    # vs B-side K handling and reveals the bit mapping per position.
    if side in ("both", "a"):
        a_codes = torch.zeros((m, k), device=device, dtype=torch.uint8)
        a_codes[0, kpos] = a_code
    else:
        a_codes = torch.full((m, k), 8, device=device, dtype=torch.uint8)
    if side in ("both", "b"):
        b_codes = torch.zeros((n, k), device=device, dtype=torch.uint8)
        b_codes[0, kpos] = b_code
    else:
        b_codes = torch.full((n, k), 12, device=device, dtype=torch.uint8)
    a = _pack_fp6_dense(a_codes)
    b = _pack_fp6_dense(b_codes)
    a_scale = torch.full((m, k // 32), 127, device=device,
                         dtype=torch.uint8).view(torch.float8_e8m0fnu)
    b_scale = torch.full((n, k // 32), 127, device=device,
                         dtype=torch.uint8).view(torch.float8_e8m0fnu)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    actual = mm_mxfp6(a, a_scale, b, b_scale, out=out, space=space).float()
    rest = actual.clone()
    if side == "a":
        # C[0,:] must be 1, everything else 0.
        row0 = actual[0].clone()
        rest[0, :] = 0
        print(f"probeA k={kpos:3d} code={a_code:2d}: C[0,0]={row0[0].item():10.6f} "
              f"C[0,1]={row0[1].item():10.6f} max|rest|={rest.abs().max().item():10.6f} "
              f"(expect 1.0 / 1.0 / 0.0)")
    elif side == "b":
        # C[:,0] must be 1, everything else 0.
        col0 = actual[:, 0].clone()
        rest[:, 0] = 0
        print(f"probeB k={kpos:3d}: C[0,0]={col0[0].item():8.3f} "
              f"C[1,0]={col0[1].item():8.3f} max|rest|={rest.abs().max().item():8.3f} "
              f"(expect 1.0 / 1.0 / 0.0)")
    else:
        rest[0, 0] = 0
        print(f"probe k={kpos:3d}: C[0,0]={actual[0, 0].item():8.3f} "
              f"max|rest|={rest.abs().max().item():8.3f} (expect 1.0 / 0.0)")


def _check_ones(m, n, k, device, space, *, negate_b=False):
    # All-ones data (e2m3 code 8, e3m2 code 12) with unit scales (E8M0 127):
    # every output must equal +/-K exactly.
    b_code = 12 | 32 if negate_b else 12
    expect = -k if negate_b else k
    a = _pack_fp6_dense(torch.full((m, k), 8, device=device, dtype=torch.uint8))
    b = _pack_fp6_dense(torch.full((n, k), b_code, device=device, dtype=torch.uint8))
    a_scale = torch.full((m, k // 32), 127, device=device,
                         dtype=torch.uint8).view(torch.float8_e8m0fnu)
    b_scale = torch.full((n, k // 32), 127, device=device,
                         dtype=torch.uint8).view(torch.float8_e8m0fnu)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    actual = mm_mxfp6(a, a_scale, b, b_scale, out=out, space=space)
    print(f"ones[{m},{n},{k}]neg_b={negate_b}: min={actual.float().min().item()} "
          f"max={actual.float().max().item()} mean={actual.float().mean().item()} (expect {expect})")
    torch.testing.assert_close(actual.float(),
                               torch.full((m, n), float(expect), device=device), atol=1e-1,
                               rtol=1e-2)


def _check_correct(m, n, k, device, space, compare_mxfp8=False):
    a, a_scale, b, b_scale, out = _make_inputs(m, n, k, device)
    reference = _dequantize(a, a_scale, ebits=2, bias=1) @ _dequantize(b, b_scale, ebits=3,
                                                                       bias=3).T
    actual = mm_mxfp6(a, a_scale, b, b_scale, out=out, space=space)
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    assert actual.dtype == torch.bfloat16, actual.dtype
    err = (actual.float() - reference).abs()
    print(f"actual[0,:8]={actual.float()[0, :8].tolist()}")
    print(f"ref   [0,:8]={reference[0, :8].tolist()}")
    print(f"actual stats: min={actual.float().min().item():.3f} max={actual.float().max().item():.3f} "
          f"mean={actual.float().mean().item():.3f}")
    print(f"ref stats: min={reference.min().item():.3f} max={reference.max().item():.3f} "
          f"mean={reference.mean().item():.3f} maxerr={err.max().item():.3f}")
    torch.testing.assert_close(actual.float(), reference, atol=1e-1, rtol=1e-2)
    print(f"OK  mm_mxfp6[{m}, {n}, {k}] space={space}: max err "
          f"{(actual.float() - reference).abs().max().item():.4f}")
    if compare_mxfp8:
        a, b = _to_mxfp8(a, b)
        actual = mm_mxfp8(a, a_scale, b, b_scale, out=out, space=space)
        torch.testing.assert_close(actual.float(), reference, atol=1e-1, rtol=1e-2)
        print(f"OK  mm_mxfp8[{m}, {n}, {k}] space={space}: max err "
              f"{(actual.float() - reference).abs().max().item():.4f}")


SHAPE_SETS = {
    "small": [(1024, 1024, 1024), (4096, 1024, 4096), (4096, 4096, 4096)],
    "bench": [(1024, 1024, 256), (1536, 1536, 1024), (4096, 4096, 1024), (4096, 4096, 4096)],
}
SHAPE_SETS["full"] = list(dict.fromkeys([
    *SHAPE_SETS["small"],
    *SHAPE_SETS["bench"],
    *((shape.m, shape.n, shape.k) for shape in (*SYNTHETIC, *FOCUS.shapes("sm100"))
      if all(dim > 0 and dim % 128 == 0 for dim in (shape.m, shape.n, shape.k))),
]))


def create_benchmark(shapes, space, warmup, rep, compare_mxfp8=False):
    providers = ["tlx_mxfp6", "tlx_mxfp8"] if compare_mxfp8 else ["tlx_mxfp6"]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=shapes,
            line_arg="provider",
            line_vals=providers,
            line_names=providers,
            ylabel="TFLOPS",
            plot_name="mm-mxfp6-performance",
            args={"space": space},
        ))
    def benchmark(M, N, K, provider, space):
        device = triton.runtime.driver.active.get_active_torch_device()
        a, a_scale, b, b_scale, out = _make_inputs(M, N, K, device)
        op = mm_mxfp6
        if provider == "tlx_mxfp8":
            a, b = _to_mxfp8(a, b)
            op = mm_mxfp8
        # With space="full" the first call autotunes (slow); do_bench warmup
        # covers it so only the best config is timed.
        fn = lambda: op(a, a_scale, b, b_scale, out=out, space=space)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep,
                                                       quantiles=(0.5, 0.2, 0.8))

        def tflops(ms):
            return 2.0 * M * N * K * 1e-12 / (ms * 1e-3)

        return tflops(ms), tflops(max_ms), tflops(min_ms)

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TLX MXFP6 GEMM (E2M3 x E3M2)")
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--shapes", type=str, default="small", choices=list(SHAPE_SETS),
                        help="full includes all supported synthetic and Blackwell GEMM focus shapes")
    parser.add_argument("--space", type=str, default="full", choices=["full", "heuristic"])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--compare-mxfp8", action="store_true",
                        help="include MXFP8 with identical values and scales in benchmarks and --check")
    parser.add_argument("--check", action="store_true",
                        help="verify numerics against a dequantized reference instead of benchmarking")
    parser.add_argument("--ones", action="store_true",
                        help="all-ones input check (every output must equal K)")
    parser.add_argument("--negate-b", action="store_true",
                        help="with --ones, fill B with -1 (every output must equal -K)")
    parser.add_argument("--mxfp8-sanity", action="store_true",
                        help="run the known-good mm_mxfp8 ones check instead (harness validation)")
    parser.add_argument("--probe", action="store_true",
                        help="sweep single-element probes across K to map coverage")
    parser.add_argument("--probe-side", type=str, default="both", choices=["both", "a", "b"],
                        help="probe one side at a time (other side all ones)")
    parser.add_argument("--probe-full", action="store_true",
                        help="with --probe, sweep every K position")
    parser.add_argument("--probe-codes", action="store_true",
                        help="with --probe, sweep code values at k=0,16,21")
    args = parser.parse_args()

    if not is_blackwell():
        print("Skipping benchmarks, no Blackwell GPU found.")
        raise SystemExit(0)

    if args.M is not None and args.N is not None and args.K is not None:
        shapes = [(args.M, args.N, args.K)]
    else:
        shapes = SHAPE_SETS[args.shapes]
    for m, n, k in shapes:
        assert m % 128 == 0 and n % 128 == 0 and k % 128 == 0, shapes

    if args.check or args.ones or args.mxfp8_sanity or args.probe:
        device = triton.runtime.driver.active.get_active_torch_device()
        for m, n, k in shapes:
            if args.mxfp8_sanity:
                _check_mxfp8_ones(m, n, k, device)
            elif args.probe:
                if args.probe_codes:
                    for kpos in [0, 16, 21]:
                        print(f"--- code sweep at k={kpos} side={args.probe_side} ---")
                        for code in [0, 1, 2, 4, 8, 12, 16, 32, 48, 63]:
                            _check_probe(m, n, k, device, args.space, kpos,
                                         side=args.probe_side, a_code=code, b_code=code)
                    raise SystemExit(0)
                for kpos in range(k) if args.probe_full else [
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 32, 33, 63, 64, 95, 96, 100,
                        124, 127]:
                    if kpos < k:
                        _check_probe(m, n, k, device, args.space, kpos, side=args.probe_side)
            elif args.ones:
                _check_ones(m, n, k, device, args.space, negate_b=args.negate_b)
            else:
                _check_correct(m, n, k, device, args.space, args.compare_mxfp8)
        raise SystemExit(0)

    benchmark = create_benchmark(shapes, args.space, args.warmup, args.rep, args.compare_mxfp8)
    benchmark.run(print_data=True)
