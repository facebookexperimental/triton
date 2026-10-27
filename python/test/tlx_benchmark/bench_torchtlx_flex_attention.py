"""Offline gfx950 FlexAttention comparison with max_autotune disabled."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import torch.nn.functional as F
from unittest import mock

import torch
import triton
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

MODS = ("alibi", "head_bias", "document_mask", "causal", "prefix_lm", "sliding_window", "softcap", "rel", "noop")


def document_lengths(seq, batch):
    rng = random.Random(0)
    lengths = [1] * batch
    for _ in range(batch * seq - batch):
        lengths[rng.randint(0, batch - 1)] += 1
    return lengths


def document_reference(tensors, seq, batch):
    pieces = []
    start = 0
    for length in document_lengths(seq, batch):
        pieces.append(F.scaled_dot_product_attention(*[t[:, :, start:start + length] for t in tensors], is_causal=True))
        start += length
    return torch.cat(pieces, dim=2)


def inputs(mod, seq, batch=8, heads=16, dim=128):
    torch.manual_seed(42)
    score_mod = None
    mask_mod = None
    opts = {}
    shape = (batch, heads, seq, dim)
    if mod in ("causal", "alibi", "softcap"):
        mask_mod = lambda b, h, m, n: m >= n
    if mod == "rel":
        score_mod = lambda s, b, h, m, n: s + (m - n)
    elif mod == "head_bias":
        score_mod = lambda s, b, h, m, n: s + 2 * h
    elif mod == "alibi":
        score_mod = lambda s, b, h, m, n: s + (n - m) * torch.exp2(-((h + 1) * 8.0 / heads))
    elif mod == "softcap":
        score_mod = lambda s, b, h, m, n: 20 * torch.tanh(s / 20)
    elif mod == "prefix_lm":
        mask_mod = lambda b, h, m, n: (n < 128) | (m >= n)
    elif mod == "sliding_window":
        mask_mod = lambda b, h, m, n: (m - n <= 4096) & (m >= n)
    elif mod == "document_mask":
        lengths = document_lengths(seq, batch)
        offsets = torch.tensor([0] + lengths, device="cuda", dtype=torch.int32).cumsum(0)
        docs = torch.repeat_interleave(torch.arange(batch, device="cuda", dtype=torch.int32),
                                       torch.tensor(lengths, device="cuda"))

        def mask_mod(b, h, m, n):
            return (docs[m] == docs[n]) & (m - offsets[docs[m]] >= n - offsets[docs[n]])

        shape = (1, heads, batch * seq, dim)
        opts = {"BLOCK_M": 128, "BLOCK_N": 32, "fwd_num_warps": 8, "fwd_num_stages": 4}
    block_mask = None
    if mask_mod is not None:
        with config.patch({"triton.tlx_mode": None}):
            block_mask = torch.compile(create_block_mask)(mask_mod, 1, 1, shape[2], shape[2], device="cuda")
    q, k, v = [torch.rand(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    return (q, k, v), score_mod, block_mask, opts


def compile_arm(tensors, score_mod, block_mask, opts, mode, out, offline=None):
    from triton.language.extra.tlx.inductor import flex_attention_templates as templates
    template = templates.gfx950_flex_attention_template
    original = template.maybe_append_choice

    def append(**kwargs):
        kwargs.update(offline)
        return original(**kwargs)

    patch = mock.patch.object(template, "maybe_append_choice",
                              append) if offline is not None and mode else contextlib.nullcontext()

    def fn(q, k, v):
        return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask, kernel_options=opts)

    from torch._inductor.codecache import PyCodeCache
    modules = []
    original_load = PyCodeCache.load_by_key_path

    def load(*args, **kwargs):
        module = original_load(*args, **kwargs)
        modules.append(module)
        return module

    torch._dynamo.reset()
    with mock.patch.object(PyCodeCache, "load_by_key_path", load), patch, config.patch(
        {"triton.tlx_mode": mode, "max_autotune": False, "max_autotune_gemm": False, "force_disable_caches": True}):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        result, sources = run_and_get_code(compiled, *tensors)
        # Capture before Dynamo reset can invalidate the selected TLX mode.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(*tensors)
        graph.replay()
        result = captured.clone()
    source = "\n".join(sources)
    out.write_text(source)
    if ("tlx_gfx950_flex_attention" in source) != bool(mode):
        raise RuntimeError(f"Unexpected template selection in {out}")
    # HIP graphs need modules and static buffers kept alive across Dynamo resets.
    def replay(_keepalive=(modules, tensors, captured, compiled)):
        graph.replay()
        return captured

    return replay, result


def time_graph(fn, rep=200):
    """Cold-L2 timing; bound live HIP events to avoid failures on tiny kernels."""
    di = triton.runtime.driver.active.get_device_interface()
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    starts = [di.Event(enable_timing=True) for _ in range(32)]
    ends = [di.Event(enable_timing=True) for _ in range(32)]
    for _ in range(100):
        fn()
    di.synchronize()
    samples = []
    while len(samples) < 96 or sum(samples) < rep:
        for start, end in zip(starts, ends):
            triton.runtime.driver.active.clear_cache(cache)
            start.record()
            fn()
            end.record()
        di.synchronize()
        samples.extend(start.elapsed_time(end) for start, end in zip(starts, ends))
    return statistics.median(samples)


def plot_results(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    seqs = sorted({r["seq"] for r in rows})
    mods = [m for m in MODS if any(r["mod"] == m for r in rows)]
    grid = np.full((len(mods), len(seqs)), np.nan)
    lookup = {(r["mod"], r["seq"]): r for r in rows}
    for i, mod in enumerate(mods):
        for j, seq in enumerate(seqs):
            grid[i, j] = lookup.get((mod, seq), {}).get("ratio", np.nan)
    fig, ax = plt.subplots(figsize=(max(7, len(seqs) * 1.1), max(3, len(mods) * 0.6)))
    im = ax.imshow(np.log2(grid), cmap="RdYlGn", vmin=-0.7, vmax=0.7, aspect="auto")
    for i, mod in enumerate(mods):
        for j, seq in enumerate(seqs):
            row = lookup.get((mod, seq), {})
            label = f"{grid[i,j]:.2f}" if np.isfinite(grid[i, j]) else "FAIL"
            if row.get("accuracy_vs_stock") is False:
                label += "†"
            ax.text(j, i, label, ha="center", va="center", fontsize=10)
    ax.set_xticks(range(len(seqs)), labels=seqs)
    ax.set_yticks(range(len(mods)), labels=mods)
    ax.set_xlabel("Sequence length per original batch entry")
    ratios = [r["ratio"] for r in rows if "ratio" in r]
    gm = math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else float("nan")
    ax.set_title(f"gfx950 flex forward: TLX / stock TFLOPS ({output.name})\nGeomean {gm:.3f}×; max_autotune off")
    fig.colorbar(im, ax=ax, label="log₂(TLX / stock)")
    fig.text(0.01, 0.01, "† Stock fails accuracy; TLX checked against per-document SDPA.", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(output / "ratios.png", dpi=170)
    plt.close(fig)


def install_flex_adapter():
    # Install before both arms; this wheel lacks the general registry's tma_inner_dim.
    from torch._inductor.choices import InductorChoices
    from torch._inductor.runtime import triton_heuristics
    from triton.language.extra.tlx.inductor.flex_attention_templates import append_tlx_flex

    class FlexChoices(InductorChoices):

        @staticmethod
        def uuid():
            return "gfx950-flex-offline-v1"

        def append_flex_attention_choices(self, choices, configs, input_nodes, subgraphs, layout, kernel_options,
                                          sparse_q_block_size, sparse_kv_block_size):
            return append_tlx_flex(choices, configs, input_nodes, subgraphs, layout, kernel_options,
                                   sparse_q_block_size, sparse_kv_block_size)

    config.inductor_choices_class = FlexChoices
    triton_heuristics.tlx_only_cuda_options = lambda: []
    original_meta = triton_heuristics.CachingAutotuner._create_compile_meta

    def compile_meta(self, cfg):
        meta = original_meta(self, cfg)
        opts = self.inductor_meta.get("config_args", {})
        backend = dict(meta.get("backend_options", {}))
        for name in ("matrix_instr_nonkdim", "waves_per_eu", "kpack"):
            if name in opts:
                backend[name] = opts[name]
        meta["backend_options"] = backend
        return meta

    triton_heuristics.CachingAutotuner._create_compile_meta = compile_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, nargs="+", default=[2**i for i in range(7, 15)])
    parser.add_argument("--mod-type", nargs="+", choices=MODS, default=MODS)
    parser.add_argument("--config", help="BLOCK_M,BLOCK_N,num_warps,waves_per_eu,matrix_instr_nonkdim,kpack")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--plot", action="store_true", help="Write one figure for the entire run")
    parser.add_argument("--stock-document-default", action="store_true",
                        help="Ablate the tritonbench document config override")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    out = args.output.resolve()
    if not out.is_relative_to(root):
        parser.error("--output must be inside this Git repository")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(out / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(out / "triton")
    os.environ["TRITON_DUMP_DIR"] = str(out / "dump")
    os.environ["TMPDIR"] = str(out / "tmp")
    (out / "tmp").mkdir(exist_ok=True)
    if triton.runtime.driver.active.get_current_target().arch != "gfx950":
        raise RuntimeError("This benchmark is MI350X/gfx950 only")
    config.compile_threads = 1
    install_flex_adapter()
    offline = None
    if args.config:
        names = ("BLOCK_M", "BLOCK_N", "num_warps", "waves_per_eu", "matrix_instr_nonkdim", "kpack")
        vals = list(map(int, args.config.split(",")))
        if len(vals) != len(names):
            parser.error("--config requires six integers")
        offline = dict(zip(names, vals), num_stages=1)
    rows = []
    for mod in args.mod_type:
        for seq in args.seq_len:
            row = {"mod": mod, "seq": seq, "config": offline}
            try:
                tensors, score_mod, block_mask, opts = inputs(mod, seq, args.batch, args.heads, args.dim)
                if args.stock_document_default and mod == "document_mask":
                    opts = {}
                ref, ref_result = compile_arm(tensors, score_mod, block_mask, opts, None, out / f"{mod}-{seq}-stock.py")
                tlx, result = compile_arm(tensors, score_mod, block_mask, opts, "force", out / f"{mod}-{seq}-tlx.py",
                                          offline)
                if mod == "document_mask":
                    independent = document_reference(tensors, seq, args.batch)
                    row["stock_sdpa_max_abs"] = (independent.float() - ref_result.float()).abs().max().item()
                    row["tlx_sdpa_max_abs"] = (independent.float() - result.float()).abs().max().item()
                try:
                    torch.testing.assert_close(result, ref_result, atol=2e-2, rtol=2e-2)
                    row["accuracy_vs_stock"] = True
                except AssertionError:
                    row["accuracy_vs_stock"] = False
                    if mod != "document_mask":
                        raise
                if mod == "document_mask":
                    torch.testing.assert_close(result, independent, atol=2e-2, rtol=2e-2)
                    row["stock_accuracy_vs_sdpa"] = row["stock_sdpa_max_abs"] < 0.02
                row["accuracy"] = True
                row["max_abs_error"] = (result.float() - ref_result.float()).abs().max().item()
                pairs = []
                for i in range(3):
                    funcs = (ref, tlx) if i % 2 == 0 else (tlx, ref)
                    times = [time_graph(f, rep=args.rep) for f in funcs]
                    pairs.append(times if i % 2 == 0 else times[::-1])
                row["stock_ms"] = sorted(p[0] for p in pairs)[1]
                row["tlx_ms"] = sorted(p[1] for p in pairs)[1]
                row["ratio"] = row["stock_ms"] / row["tlx_ms"]
                row["pairs_ms"] = pairs
                flops = 4 * args.batch * args.heads * seq * seq * args.dim
                row["tlx_tflops"] = flops / row["tlx_ms"] / 1e9
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)
            (out / "results.json").write_text(
                json.dumps({"host": os.uname().nodename, "args": vars(args) | {"output": str(out)}, "rows": rows},
                           indent=2))
            print(json.dumps(row), flush=True)
    ratios = [r["ratio"] for r in rows if "ratio" in r]
    if ratios:
        print("geomean", math.exp(sum(map(math.log, ratios)) / len(ratios)), flush=True)
    if args.plot:
        plot_results(rows, out)
    if any("error" in r for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
