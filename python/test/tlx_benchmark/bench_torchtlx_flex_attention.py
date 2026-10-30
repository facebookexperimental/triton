"""TorchTLX FlexAttention forward comparison with max_autotune disabled."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
import random
import re
import torch.nn.functional as F
from unittest import mock
from typing import NamedTuple
from uuid import uuid4

import torch
import triton
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from triton.tlx.ops.kernels._shape_suites import FocusRegistry, FocusSuite

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import Case, Prepared, driver  # noqa: E402

OP = "flex_attention_torchtlx"
REF_NAME = "torch.compile flex_attention (TLX off)"
DEFAULT_SPACE = "heuristic"
# Both graphs compile during prepare; replay cannot provide a cold compile pass.
COLD_COMPILE = "none"
ARTIFACT_DIR = Path(__file__).resolve().parents[3] / ".cache/tlx_benchmark"
EXTRA_COLUMNS = (("stock accurate", "stock_accuracy"), ("TLX error", "max_abs_error"))

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


def inputs(mod, seq, batch=8, heads=16, dim=128, dtype=torch.bfloat16):
    torch.manual_seed(42)
    score_mod = None
    mask_mod = None
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
    elif mod == "strided":
        # Odd 128-blocks lose their upper half: both partial and full KV block lists have gaps.
        mask_mod = lambda b, h, m, n: (m >= n) & ((n // 64) % 4 != 3)
    elif mod == "document_mask":
        lengths = document_lengths(seq, batch)
        offsets = torch.tensor([0] + lengths, device="cuda", dtype=torch.int32).cumsum(0)
        docs = torch.repeat_interleave(torch.arange(batch, device="cuda", dtype=torch.int32),
                                       torch.tensor(lengths, device="cuda"))

        def mask_mod(b, h, m, n):
            return (docs[m] == docs[n]) & (m - offsets[docs[m]] >= n - offsets[docs[n]])

        shape = (1, heads, batch * seq, dim)
    block_mask = None
    if mask_mod is not None:
        with config.patch({"triton.tlx_mode": None}):
            block_mask = torch.compile(create_block_mask)(mask_mod, 1, 1, shape[2], shape[2], device="cuda")
    q, k, v = [torch.rand(shape, device="cuda", dtype=dtype) for _ in range(3)]
    return (q, k, v), score_mod, block_mask


def compile_arm(tensors, score_mod, block_mask, mode, out):

    def fn(q, k, v):
        return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

    from torch._inductor.codecache import PyCodeCache
    modules = []
    original_load = PyCodeCache.load_by_key_path

    def load(*args, **kwargs):
        module = original_load(*args, **kwargs)
        modules.append(module)
        return module

    torch._dynamo.reset()
    with mock.patch.object(PyCodeCache, "load_by_key_path", load), config.patch(
        {"triton.tlx_mode": mode, "max_autotune": False, "max_autotune_gemm": False, "force_disable_caches": True}):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        _, sources = run_and_get_code(compiled, *tensors)
        # Capture before Dynamo reset can invalidate the selected TLX mode.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(*tensors)
        graph.replay()
        result = captured.clone()
    source = "\n".join(sources)
    out.write_text(source)
    if bool(re.search(r"\btlx\.\w+\(", source)) != bool(mode):
        raise RuntimeError(f"Unexpected template selection in {out}")
    # Graphs need modules and static buffers kept alive across Dynamo resets.
    def replay(_keepalive=(modules, tensors, captured, compiled)):
        graph.replay()
        return captured

    return replay, result


def plot_results(rows, output, arch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = list(dict.fromkeys(row["group"] for row in rows))
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 6 * len(groups)), squeeze=False, constrained_layout=True)
    for group, ax in zip(groups, axes[:, 0]):
        entries = [row for row in rows if row["group"] == group]
        seqs = sorted({row["seq"] for row in entries})
        mods = [mod for mod in MODS if any(row["mod"] == mod for row in entries)]
        lookup = {(row["mod"], row["seq"]): row for row in entries}
        grid = np.full((len(mods), len(seqs)), np.nan)
        for i, mod in enumerate(mods):
            for j, seq in enumerate(seqs):
                grid[i, j] = lookup.get((mod, seq), {}).get("ratio", np.nan)
        im = ax.imshow(np.log2(grid), cmap="RdYlGn", vmin=-0.7, vmax=0.7, aspect="auto")
        for i, mod in enumerate(mods):
            for j, seq in enumerate(seqs):
                row = lookup.get((mod, seq))
                if row is None:
                    continue
                label = f"{grid[i, j]:.2f}" if np.isfinite(grid[i, j]) else "FAIL"
                if row.get("stock_accuracy") is False:
                    label = "N/A†"
                ax.text(j, i, label, ha="center", va="center")
        ax.set_xticks(range(len(seqs)), labels=seqs)
        ax.set_yticks(range(len(mods)), labels=mods)
        ax.set_xlabel("Sequence length per original batch entry")
        ax.set_title(group)
        fig.colorbar(im, ax=ax, label="log₂(TLX / stock)")
    ratios = [row["ratio"] for row in rows if math.isfinite(row["ratio"])]
    gm = math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else float("nan")
    fig.suptitle(f"{arch} flex forward: TLX / stock TFLOPS; geomean {gm:.3f}×; max_autotune off\n"
                 "† Stock fails accuracy; no ratio reported. TLX checked against per-document SDPA.")
    fig.savefig(output / "ratios.png", dpi=170)
    plt.close(fig)


@contextlib.contextmanager
def flex_adapter():
    # Avoid the general registry's tma_inner_dim dependency on older torch wheels.
    from torch._inductor.runtime import triton_heuristics
    from triton.language.extra.tlx.inductor.choices import TLXInductorChoices

    original_meta = triton_heuristics.CachingAutotuner._create_compile_meta

    def compile_meta(self, cfg):
        meta = original_meta(self, cfg)
        if self.device_props.type == "hip":
            backend = dict(meta.get("backend_options", {}))
            opts = self.inductor_meta.get("config_args", {})
            for name in ("matrix_instr_nonkdim", "waves_per_eu", "kpack"):
                if name in opts:
                    backend[name] = opts[name]
            meta["backend_options"] = backend
        return meta

    with (
            mock.patch.object(triton_heuristics, "tlx_only_cuda_options", lambda: ["ctas_per_cga"]),
            mock.patch.object(triton_heuristics.CachingAutotuner, "_create_compile_meta", compile_meta),
            config.patch({"inductor_choices_class": TLXInductorChoices, "compile_threads": 1}),
    ):
        yield


class FlexAttentionShape(NamedTuple):
    batch: int
    heads: int
    seq: int
    dim: int
    mod: str
    dtype: str = "bfloat16"


SHAPE_SUITES = FocusRegistry(
    OP,
    tuple(FocusSuite(mod, OP, tuple(FlexAttentionShape(8, 16, 2**i, 128, mod) for i in range(7, 15))) for mod in MODS) +
    (
        FocusSuite("builtin", OP, includes=MODS),
        FocusSuite("fp16", OP,
                   tuple(FlexAttentionShape(8, 16, 2**i, 128, mod, "float16") for mod in MODS for i in range(7, 15))),
    ),
    {"gfx950": ("builtin", ), "sm100": ("builtin", )},
)
SYNTHETIC = tuple(FlexAttentionShape(1, 2, seq, dim, mod) for mod in MODS for seq, dim in ((128, 64), (256, 128)))


def shapes(synthetic=False, suites=None):
    return list(SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites))


def cases(synthetic=False, suites=None):
    return [
        Case(OP, driver.arch(), entry.dtype, tuple(entry[:-1]),
             label=f"{entry.mod} B={entry.batch} H={entry.heads} S={entry.seq} D={entry.dim} {entry.dtype}")
        for entry in shapes(synthetic, suites)
    ]


def prepare(case, space):
    if space != "heuristic":
        raise ValueError("TorchTLX flex uses offline heuristic configs; max_autotune stays disabled")
    root = ARTIFACT_DIR / f"flex_attention.{driver.arch()}.{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    cache_env = {
        name: str(root / subdir)
        for name, subdir in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"),
                             ("TRITON_DUMP_DIR", "dump"), ("TMPDIR", "tmp"))
    }
    for directory in cache_env.values():
        Path(directory).mkdir(exist_ok=True)
    with mock.patch.dict(os.environ, cache_env), mock.patch.object(triton.knobs.cache, "dir",
                                                                   cache_env["TRITON_CACHE_DIR"]), flex_adapter():
        entry = FlexAttentionShape(*case.shape, dtype=case.dtype)
        tensors, score_mod, block_mask = inputs(entry.mod, entry.seq, entry.batch, entry.heads, entry.dim,
                                                dtype=getattr(torch, entry.dtype))
        name = case.key.replace("/", "_")
        ref, ref_result = compile_arm(tensors, score_mod, block_mask, None, root / f"{name}.stock.py")
        tlx, result = compile_arm(tensors, score_mod, block_mask, "force", root / f"{name}.tlx.py")
    reference = document_reference(tensors, entry.seq, entry.batch) if entry.mod == "document_mask" else ref_result
    stock_accuracy, _ = _accuracy(ref_result, reference)
    extra = {"stock_accuracy": stock_accuracy, "max_abs_error": (result.float() - reference.float()).abs().max().item()}
    if not stock_accuracy:
        # An inaccurate stock result cannot support a performance ratio.
        ref = None
    return Prepared(tlx_fn=tlx, ref_fn=ref, flop_count=4 * entry.batch * entry.heads * entry.seq**2 * entry.dim,
                    check=lambda: _accuracy(tlx(), reference), extra=extra)


def _accuracy(result, reference):
    try:
        torch.testing.assert_close(result, reference, atol=2e-2, rtol=2e-2)
    except AssertionError as exc:
        return False, str(exc)
    return True, ""


def annotate(result):
    if result.extra.get("stock_accuracy") is False:
        result.notes.append("Stock fails per-document SDPA accuracy; no stock performance ratio reported")


_, default_json, run, _main = driver.bind(sys.modules[__name__])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--json")
    parser.add_argument("--list-suites", action="store_true")
    parser.add_argument("--list-suite")
    args, _ = parser.parse_known_args(argv)
    if args.list_suites or args.list_suite:
        return _main(argv)
    path = Path(args.json) if args.json else ARTIFACT_DIR / f"{OP}.{uuid4().hex[:8]}" / "results.json"
    if args.json is None:
        argv.extend(["--json", str(path)])
    previous_mtime = path.stat().st_mtime_ns if path.exists() else None
    status = _main(argv)
    if not path.exists() or path.stat().st_mtime_ns == previous_mtime:
        return status
    artifact = json.loads(path.read_text())
    results = artifact["results"]
    if not results:
        return status
    output = path.parent / path.stem
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        case = result["case"]
        batch, heads, seq, dim, mod = case["shape"]
        valid = result["status"] != "error" and result["speedup"] is not None
        rows.append({
            "mod": mod, "seq": seq, "ratio": result["speedup"] if valid else float("nan"), "stock_accuracy":
            result["extra"].get("stock_accuracy"), "group": f"B={batch} H={heads} D={dim} {case['dtype']}"
        })
    plot_results(rows, output, results[0]["case"]["arch"])
    artifact["env"]["plot"] = str(output / "ratios.png")
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"plot: {artifact['env']['plot']}")
    return status


def supported():
    return driver.arch() in SHAPE_SUITES.defaults


if __name__ == "__main__":
    raise SystemExit(main())
