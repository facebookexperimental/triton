"""Checks that curated autotune spaces still keep up with the grids they were picked from.

On gfx950, `tlx.ops.hstu_cross_attn` autotunes its forward at head dims 64 and
128 over eight curated configs (`_FWD_CURATED_BY_HEAD_DIM`) picked from a
144-config grid (`space="full"`). A kernel or compiler change can move the
best config outside that list. The test tunes every gfx950 focus shape that
the tiled kernel runs over the full grid and fails when the best curated config
is more than `MAX_LOSS` slower than the grid's best.

The focus shapes that the persistent stream-K forward takes instead have no
configs to tune. A second test tunes the tiled grid on them and fails when
stream-K is slower than the grid's best, since the dispatch would then pick the
slower kernel.

A failure is a lead, not a verdict: the best of 144 noisy timings is biased low
against the best of a short list. Re-time the two configs it names before
changing a curated list, and look for the cause (kernel change, compiler
change, a new shape regime) rather than growing the list back toward the grid.

It takes about as long as tuning the grid on every shape, so it only runs when
`TLX_AUTOTUNE_DRIFT=1`.
"""

import importlib
import math
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

#: Above the run-to-run difference of an autotune pass on these shapes (~1-3%).
MAX_LOSS = 0.05

pytestmark = pytest.mark.skipif(
    os.environ.get("TLX_AUTOTUNE_DRIFT") != "1", reason="slow; set TLX_AUTOTUNE_DRIFT=1 to run")


def _label(shape):
    return f"{shape.model} z={shape.batch} sm={shape.num_softmax_heads}"


def _stream_k_labels(monkeypatch):
    """Labels of the gfx950 focus shapes that the forward dispatch sends to stream-K."""
    import torch
    from triton.tlx.ops import hstu_cross_attn
    from triton.tlx.ops.kernels.hstu_cross_attn._shapes import FOCUS, inputs

    xa = importlib.import_module("triton.tlx.ops.kernels.hstu_cross_attn.gfx950")
    stream_k_fwd = xa.stream_k_fwd
    calls = []
    with monkeypatch.context() as m:
        m.setenv(xa.STREAM_K_ENV, "1")
        m.setattr(xa, "stream_k_fwd", lambda *a: calls.append(1) or stream_k_fwd(*a))
        labels = set()
        for shape in FOCUS.shapes("gfx950"):
            args = inputs(shape)
            args.pop("lengths_q"), args.pop("lengths_kv")
            calls.clear()
            with torch.inference_mode():
                # A shape the tiled kernel takes tunes a single config here.
                hstu_cross_attn(space="smoke", **args)
            if calls:
                labels.add(_label(shape))
    return labels


def _check_curated(monkeypatch, kernels, autotuners, call, config_key, curated_for, curated_name, skip=frozenset()):
    """Tune every gfx950 focus shape not labelled in `skip` with `call(args)` and compare each tuning pass of
    `kernels`.

    The autotune keys bucket batch and lengths, so several focus shapes share a key. `autotuners`
    are cleared before each shape so every shape gets its own tuning pass; compiled kernels stay.
    """
    from _harness import driver

    # Fail rather than skip: a skipped weekly run would read as a pass.
    assert driver.arch() == "gfx950", f"TLX_AUTOTUNE_DRIFT=1 but the GPU is {driver.arch()}, not gfx950"
    import torch
    from triton import knobs
    from triton.tlx.ops.kernels.hstu_cross_attn._shapes import FOCUS, inputs

    passes = []  # (shape label, kernel, key, {config key: median ms}, cache_hit, curated list)
    current = {}

    def listener(fn, key, best_config, configs_timings, duration, cache_hit):
        name = kernels.get(fn.__name__)
        if name is not None:
            # Timings are [median, p20, p80]; compare medians.
            timings = {config_key(c): t[0] for c, t in (configs_timings or {}).items()}
            passes.append((current["shape"], name, key, timings, cache_hit, current["curated"]))

    monkeypatch.setattr(knobs.autotuning, "listener", listener)
    labels = []
    for shape in FOCUS.shapes("gfx950"):
        if _label(shape) in skip:
            continue
        current["shape"] = _label(shape)
        current["curated"] = curated_for(shape)
        assert current["curated"], f"{current['shape']}: head dim {shape.head_dim} has no curated list to check"
        labels.append(current["shape"])
        args = inputs(shape)
        args.pop("lengths_q"), args.pop("lengths_kv")
        for at in autotuners:
            at.cache.clear()
        call(args)
        torch.cuda.synchronize()

    assert not [p for p in passes if p[4]], "autotune results came from the disk cache, not a fresh measurement"
    assert {p[1] for p in passes} == set(kernels.values()), f"not every kernel was tuned: {passes}"
    untuned = sorted(set(labels) - {p[0] for p in passes})
    assert not untuned, f"focus shapes with no tuning pass of their own: {untuned}"
    rows, drifted = [], []
    for label, name, key, timings, _, curated in passes:
        valid = {k: t for k, t in timings.items() if math.isfinite(t) and t > 0}
        in_list = {k: t for k, t in valid.items() if k in curated}
        assert in_list, f"{label} {name}: no curated config produced a timing ({timings})"
        best_key = min(valid, key=valid.get)
        best_curated = min(in_list, key=in_list.get)
        loss = in_list[best_curated] / valid[best_key] - 1
        rows.append(f"{label} {name}: {100 * loss:+.1f}%")
        if loss > MAX_LOSS:
            drifted.append(f"{rows[-1]} (key {key}): grid best {best_key} {valid[best_key]:.4f} ms, "
                           f"curated best {best_curated} {in_list[best_curated]:.4f} ms")
    print(f"\n{len(passes)} tuning passes\n" + "\n".join(rows))
    assert not drifted, (f"{curated_name} fell behind its full grid; re-time these before changing it "
                         "(BLOCK_M, BLOCK_N, NUM_STAGES, matrix_instr_nonkdim, waves_per_eu, num_warps):\n" +
                         "\n".join(drifted))


def test_hstu_cross_attn_curated_forward_space_tracks_the_full_grid(governor, monkeypatch):
    from triton.tlx.ops import hstu_cross_attn

    xa = importlib.import_module("triton.tlx.ops.kernels.hstu_cross_attn.gfx950")
    _check_curated(
        monkeypatch,
        {"_tlx_gfx950_cross_attn_fwd": "fwd"},
        [xa._tlx_gfx950_cross_attn_fwd.full],
        lambda args: hstu_cross_attn(space="full", **args),
        xa._fwd_config_key,
        lambda shape: xa._FWD_CURATED_BY_HEAD_DIM.get(shape.head_dim),
        "_FWD_CURATED_BY_HEAD_DIM",
        skip=_stream_k_labels(monkeypatch),
    )


def test_hstu_cross_attn_stream_k_beats_the_tiled_grid(governor, monkeypatch):
    import torch
    from triton import knobs
    from triton.tlx.ops import hstu_cross_attn
    from triton.tlx.ops.kernels.hstu_cross_attn._shapes import FOCUS, inputs

    xa = importlib.import_module("triton.tlx.ops.kernels.hstu_cross_attn.gfx950")
    full = xa._tlx_gfx950_cross_attn_fwd.full
    labels = _stream_k_labels(monkeypatch)
    assert labels, "no gfx950 focus shape takes stream-K"
    passes = []

    def listener(fn, key, best_config, configs_timings, duration, cache_hit):
        passes.append((configs_timings, cache_hit))

    monkeypatch.setattr(knobs.autotuning, "listener", listener)
    rows, slower = [], []
    for shape in FOCUS.shapes("gfx950"):
        if _label(shape) not in labels:
            continue
        args = inputs(shape)
        args.pop("lengths_q"), args.pop("lengths_kv")
        with torch.inference_mode():
            monkeypatch.setenv(xa.STREAM_K_ENV, "1")
            # The grid's own benchmarker, on the whole op: stream-K's host work counts against it.
            stream_k_ms = full.do_bench(lambda: hstu_cross_attn(**args), quantiles=(0.5, 0.2, 0.8))[0]
            monkeypatch.setenv(xa.STREAM_K_ENV, "0")
            full.cache.clear()
            passes.clear()
            hstu_cross_attn(space="full", **args)
            torch.cuda.synchronize()
        assert len(passes) == 1 and not passes[0][1], f"{_label(shape)}: expected one fresh tuning pass, {passes}"
        valid = {xa._fwd_config_key(c): t[0] for c, t in passes[0][0].items() if math.isfinite(t[0]) and t[0] > 0}
        best_key = min(valid, key=valid.get)
        loss = stream_k_ms / valid[best_key] - 1
        rows.append(f"{_label(shape)}: stream-K {stream_k_ms:.4f} ms, grid best {best_key} {valid[best_key]:.4f} ms "
                    f"({100 * loss:+.1f}%)")
        if loss > 0:
            slower.append(rows[-1])
    print("\n" + "\n".join(rows))
    assert not slower, "stream-K is slower than the tiled grid's best, so the dispatch picks the slower kernel:\n" + \
        "\n".join(slower)
