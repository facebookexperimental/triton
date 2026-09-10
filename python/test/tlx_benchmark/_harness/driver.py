"""The part of a benchmark that is the same for every op.

`bench_<op>.py` supplies what only it can know -- the shapes, how to build the
operands, what to race against, how many FLOPs that is -- and nothing else.
Device selection, clock governing, the cold-compile pass, the measurement
window, the verdict, the table and the artifact all live here, so adding an op
does not mean copying a CLI.

The adapter is the bench module itself, duck-typed, because `test_ops_perf.py`
already discovers and calls bench modules that way. Five names:

    OP              str                       -- catalog op name
    REF_NAME        str                       -- what `ref_fn` is; lands in env["ref"]
    EXTRA_COLUMNS   ((header, key), ...)      -- which Result.extra keys get a column
    cases(synthetic)            -> list[Case]
    prepare(case, space)        -> Prepared

and one line of wiring:

    supported, default_json, run, main = driver.bind(sys.modules[__name__])
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import os
from typing import Any, Callable, Iterable, Optional

from . import report as report_mod
from . import verdict
from .compile import COLD_COMPILE_CAP_S, cold_compile
from .contract import Case, Result, Status
from .denoise import Governor, capture_env, list_devices, select_device, stable
from .measure import DEFAULT_REPLICATES, LATENCY_MODES, host_overhead_us, measure


@dataclasses.dataclass
class Prepared:
    """One case's operands, closures and cost model, from `bench.prepare`.

    The closures must not allocate: they are called several hundred times inside
    the measured window, and an allocation there is timed as if it were the
    kernel. Build the tensors in `prepare` and capture them.
    """

    #: Called in the measured window. For a backward case this is the
    #: `.backward()` call, not the forward.
    tlx_fn: Callable[[], Any]
    #: What TLX is raced against, or None when the op has no runnable reference
    #: -- then `speedup` is not reported and `floor_tflops` is the only perf gate.
    ref_fn: Optional[Callable[[], Any]] = None
    #: Useful FLOPs for one call of `tlx_fn`. What makes every op's numbers
    #: commensurable, since the suite reports throughput rather than latency.
    flop_count: int = 0
    #: Tensors whose `.grad` must be cleared between timed iterations. Required
    #: for a backward case, where otherwise each iteration accumulates into the
    #: previous one's gradients.
    grad_to_none: Optional[Iterable] = None
    #: Absolute TFLOP/s gate, for an op with no reference. See `verdict.judge`.
    floor_tflops: Optional[float] = None
    #: `() -> (ok, note)`. The op owns this because only it knows what its output
    #: is and what tolerance applies; see `close_enough`. None skips the check --
    #: appropriate for a backward case, whose numerics L1 already covers.
    check: Optional[Callable[[], tuple]] = None
    #: Op-specific derived metrics. See `Result.extra`.
    extra: dict = dataclasses.field(default_factory=dict)
    #: Cold-compile ceiling for this case. Overridable per op because an op with
    #: no `heuristic_config` autotunes a full space on its first call and will
    #: legitimately exceed the default; see the suite README.
    cap_s: float = COLD_COMPILE_CAP_S


#: The GPU this run is about. Everything downstream -- which `PERF_SHAPES` are
#: imported, which `arch=` the op is pinned to, which device's clocks are
#: captured, what the artifact is named -- has to agree with it.
#:
#: Set once by `select`, from `main`'s `--device`. Absent that (the pytest
#: entry point, which does no selection) it falls back to the first device,
#: matching what torch will call `cuda:0`. It cannot be derived from
#: `list_devices()[0]` unconditionally: `main` may pin GPU N and set the
#: visibility variable, and `nvidia-smi` enumerates physical devices regardless
#: of that, so on a heterogeneous host GPU 0's arch is simply the wrong answer.
_SELECTED: list = []  # a one-slot box, so "unset" and "no GPU" stay distinct


def select(device) -> None:
    _SELECTED[:] = [device]


def selected_device():
    if not _SELECTED:
        devices = list_devices()
        _SELECTED.append(devices[0] if devices else None)
    return _SELECTED[0]


def arch() -> Optional[str]:
    device = selected_device()
    return device.arch if device else None


def device_index() -> int:
    """Physical index, for the `nvidia-smi`-backed denoise helpers.

    Physical rather than CUDA-visible: those helpers shell out to `nvidia-smi`,
    which enumerates every GPU and does not honour the visibility variable.
    """
    device = selected_device()
    return device.index if device else 0


def close_enough(out, ref, rel: float) -> tuple:
    """`(ok, note)` for the common "one tensor against one tensor" check.

    atol tracks the magnitude of the whole result rather than of the element
    being compared, which is what the L1 suites do, so a case cannot pass there
    and fail here.
    """
    import torch

    try:
        torch.testing.assert_close(out, ref, atol=rel * ref.abs().max().item(), rtol=rel)
    except AssertionError as mismatch:
        return False, f"output does not match the reference: {str(mismatch).splitlines()[0]}"
    return True, ""


#: Kwarg-name prefixes worth dropping from the reported config: the kernels
#: spell the tile either BLOCK_M or BLOCK_SIZE_M, and neither prefix carries
#: information once the column is called "best config".
_CONFIG_ABBREV = (("BLOCK_SIZE_", "B"), ("BLOCK_", "B"))


def _fmt_config(config) -> str:
    parts = []
    for key, value in config.kwargs.items():
        for long, short in _CONFIG_ABBREV:
            if key.startswith(long):
                key = short + key[len(long):]
                break
        parts.append(f"{key}={value}")
    return " ".join(parts + [f"w{config.num_warps}", f"s{config.num_stages}"])


@contextlib.contextmanager
def _record_configs(into: dict):
    """Record the config every autotuned kernel launched in this block ran with.

    Wraps `Autotuner.run` rather than using `knobs.autotuning.listener`: the
    listener only fires for a multi-config space, and a single-config one --
    what an op with a `heuristic_config` uses by default -- is the case most
    worth labelling. Doing it here rather than in a bench module is what makes
    the column op-agnostic: an op that launches several autotuned kernels (a
    flash_attn backward) contributes one entry per kernel, keyed by name.
    """
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:  # nothing to record; not a reason to fail a measurement
        yield
        return

    original = Autotuner.run

    def run(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        config = getattr(self, "best_config", None)
        if config is not None:
            into[self.base_fn.__name__] = _fmt_config(config)
        return out

    Autotuner.run = run
    try:
        yield
    finally:
        Autotuner.run = original


def _render_configs(configs: dict) -> Optional[str]:
    if not configs:
        return None  # the op's kernels are not autotuned
    if len(configs) == 1:
        return next(iter(configs.values()))
    return " | ".join(f"{name}: {config}" for name, config in configs.items())


def supported(bench) -> bool:
    # Both halves matter. No GPU means no arch; an arch the catalog has no entry
    # for means every case would raise UnsupportedOp, and N error rows read like
    # a failure when the honest answer is "not implemented here".
    from triton.tlx.ops._catalog import has_impl

    return arch() is not None and has_impl(bench.OP, arch())


def default_json(bench) -> str:
    return f"/tmp/tlx_benchmark/{bench.OP}.{arch()}.json"


def run_case(bench, case: Case, *, space: str, cold: bool = True, latency_mode: str = "wallclock") -> Result:
    import torch

    prep = bench.prepare(case, space)

    # Cold pass first, on its own fresh Triton cache. For a backward case
    # `prepare` has already run the forward, so what this times is the backward
    # compile alone -- which is the number that matters, the forward's being
    # attributed to its own case.
    #
    # `cold=False` skips it and the case reports no compile time. See
    # `resolve_cold_compile` for why that is the default on some ops.
    compile_stat = cold_compile(prep.tlx_fn, cap_s=prep.cap_s) if cold else None

    configs: dict = {}
    with _record_configs(configs):
        prep.tlx_fn()  # tune and compile outside the measured window
    if prep.ref_fn is not None:
        prep.ref_fn()
    torch.cuda.synchronize()
    correct, accuracy_note = prep.check() if prep.check is not None else (None, "")

    # `measure` converts every timed iteration through flop_count, so both
    # providers come back in TFLOP/s with the dispersion measured on that
    # quantity rather than on latency.
    tlx = measure(prep.tlx_fn, flop_count=prep.flop_count, replicates=DEFAULT_REPLICATES,
                  grad_to_none=prep.grad_to_none, mode=latency_mode)
    ref = None
    if prep.ref_fn is not None:
        ref = measure(prep.ref_fn, flop_count=prep.flop_count, replicates=DEFAULT_REPLICATES,
                      grad_to_none=prep.grad_to_none, mode=latency_mode)
    host_us = host_overhead_us(prep.tlx_fn)

    result = verdict.judge(case, tlx, ref, tlx_host_us=host_us, compile_stat=compile_stat, correct=correct,
                           accuracy_note=accuracy_note, floor_tflops=prep.floor_tflops)
    result.flop_count = prep.flop_count
    result.extra = dict(prep.extra)
    result.best_config = _render_configs(configs)

    # Optional adapter hook, for an op-specific metric that can only be computed
    # from the measurement -- `prepare` runs before there is one. Mutates in
    # place; a returned value is ignored.
    annotate = getattr(bench, "annotate", None)
    if annotate is not None:
        annotate(result)

    # `prep` and everything it closes over dies with this frame; empty_cache then
    # returns the blocks to the driver so the next case can be allocated.
    del prep
    torch.cuda.empty_cache()
    return result


def _errored(case: Case, exc: Exception) -> Result:
    result = Result(case=case, status=Status.ERROR)
    result.notes.append(f"{type(exc).__name__}: {exc}")
    return result


def resolve_space(bench, space: Optional[str]) -> str:
    """`None` means "whatever this op's own default is".

    Ops do not share one: `mm` has a `heuristic_config` and defaults to it,
    while `flash_attn`/`hstu_attn_dev`/`kimi_delta_attention` have none and
    their kernels accept only "full" or "smoke" -- passing a global default of
    "heuristic" to those is a KeyError, not a slow path.
    """
    return space or getattr(bench, "DEFAULT_SPACE", "heuristic")


#: How often to pay the cold-compile pass.
#:
#: "all" is one fresh-cache first call per case. That is the honest reading, and
#: it is what an op with a heuristic should do -- mm's cold pass is under a
#: second. For an op that autotunes a full space it is the whole runtime:
#: hstu_attn compiles 48 forward configs per pass, and running it per case pays
#: that five times to answer a question that is not per-shape ("does a first
#: call take too long"). "first" samples one case per direction; the rest report
#: no compile time and are not gated on it. "none" skips it entirely.
COLD_COMPILE_MODES = ("all", "first", "none")


def resolve_cold_compile(bench, mode: Optional[str]) -> str:
    mode = mode or getattr(bench, "COLD_COMPILE", "all")
    if mode not in COLD_COMPILE_MODES:
        raise ValueError(f"cold_compile must be one of {COLD_COMPILE_MODES}, got {mode!r}")
    return mode


def _head_per_direction(cases, head: int):
    """The first `head` cases of EACH direction, in the original order.

    Per direction rather than overall because `cases()` interleaves them: a flat
    slice of an fwd/bwd list gives `head/2` of each, so `--head 10` on an op with
    a backward would quietly measure five shapes. It also keeps `--head N`
    meaning the same thing whether or not the op has a backward.
    """
    counts: dict = {}
    kept = []
    for case in cases:
        seen = counts.get(case.direction, 0)
        if seen < head:
            counts[case.direction] = seen + 1
            kept.append(case)
    return kept


def run(bench, *, space=None, head=None, synthetic=False, governor=None, cold_compile_mode=None, directions=None,
        latency_mode="wallclock"):
    space = resolve_space(bench, space)
    cold_mode = resolve_cold_compile(bench, cold_compile_mode)
    env = capture_env(device_index())
    if governor is not None:
        env["governed"] = governor.to_dict()
    cases = bench.cases(synthetic)
    if directions:
        cases = [c for c in cases if c.direction in directions]
    if head:
        cases = _head_per_direction(cases, head)
    results = []
    sampled = set()
    with stable(device_index()) as info:
        for case in cases:
            cold = cold_mode == "all" or (cold_mode == "first" and case.direction not in sampled)
            if cold:
                sampled.add(case.direction)
            try:
                results.append(run_case(bench, case, space=space, cold=cold, latency_mode=latency_mode))
            except Exception as exc:  # a broken case must not hide the others
                results.append(_errored(case, exc))
    # The autotune space is part of what a number means: a heuristic-space
    # latency and a full-space latency for the same shape differ by 4x, so two
    # artifacts are only comparable when this matches. Same for the reference --
    # not every op races torch, and a ratio against a Triton kernel is not the
    # same claim as a ratio against a vendor library.
    env["space"] = space
    env["ref"] = getattr(bench, "REF_NAME", "")
    env["cold_compile"] = cold_mode
    env["latency_mode"] = latency_mode
    if directions:
        env["directions"] = sorted(directions)
    env["replicates"] = DEFAULT_REPLICATES
    if head:
        env["head"] = head
    env["shapes"] = "synthetic" if synthetic else "focus"
    env["run"] = {k: info[k] for k in ("problems", "clock_trace", "elapsed_s") if k in info}
    return results, env


def main(bench, argv=None) -> int:
    parser = argparse.ArgumentParser(description=bench.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="GPU index, or 'auto' (default) for the least-used one")
    parser.add_argument(
        "--space", choices=("heuristic", "full", "smoke"), default=None,
        help=f"autotune search space (default {resolve_space(bench, None)}); the default is what this op "
        "uses by default, and measuring anything else measures a path users do not take")
    parser.add_argument("--head", type=int, default=None, metavar="N",
                        help="only the first N cases per direction, for a quick look")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="run the correctness shapes instead of this arch's focus list; they are "
        "mostly too small to time, so this is for looking, not for gating")
    # An op with a backward reports both by default, in two tables.
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--fwd-only", action="store_true", help="skip the backward cases")
    only.add_argument("--bwd-only", action="store_true", help="skip the forward cases")
    parser.add_argument(
        "--latency-measure-mode", choices=LATENCY_MODES, default="wallclock", dest="latency_mode",
        help="'wallclock' (default) times each call as a caller would see it, host dispatch "
        "included; 'gpu_events' pre-enqueues the batch behind a blocked stream to isolate "
        "device time, which matters most on the multi-kernel backward passes")
    parser.add_argument(
        "--cold-compile", choices=COLD_COMPILE_MODES, default=None, dest="cold_compile",
        help=f"how often to time a first call on a fresh cache (default {resolve_cold_compile(bench, None)}); "
        "'all' is per case, 'first' samples one case per direction, 'none' skips it")
    parser.add_argument("--json", default=None,
                        help="machine-readable artifact (default /tmp/tlx_benchmark/<op>.<arch>.json)")
    args = parser.parse_args(argv)
    directions = ("fwd", ) if args.fwd_only else ("bwd", ) if args.bwd_only else None

    # Pick and pin the GPU before torch touches CUDA. Selection has to happen
    # here rather than in a wrapper script so that the suite is one command,
    # and it has to happen before the first CUDA call because the visibility
    # variable is read once at context creation.
    device = select_device(args.device)
    # Pin it before anything reads `arch()`: the shape import, the `arch=` the
    # op is dispatched on, the denoise capture and the artifact name all come
    # from this one object.
    select(device)
    if device is not None:
        os.environ[device.visibility_env] = str(device.index)
        print(f"device: gpu{device.index} {device.name} "
              f"({'least used' if args.device == 'auto' else 'requested'}, "
              f"{device.memory_used_mib:.0f} MiB in use)")

    # Governing is unconditional: a number taken on an ungoverned machine is not
    # comparable to anything, so there is no switch to take one.
    with Governor(device) as governor:
        for step in governor.applied:
            print(f"  denoise: {step}")
        for step in governor.skipped:
            print(f"  denoise: SKIPPED {step}")
        results, env = run(bench, space=args.space, head=args.head, synthetic=args.synthetic, governor=governor,
                           cold_compile_mode=args.cold_compile, directions=directions, latency_mode=args.latency_mode)
    if not results:
        # An empty focus list is legitimate -- an arch may have no capture yet --
        # but a silent zero-row table reads like a pass. Say what was empty.
        what = f"no {'synthetic' if args.synthetic else 'focus'} shapes for {arch()}"
        if directions:
            what += f" with direction in {sorted(directions)}"
        print(f"{what}; nothing measured")
        return 0
    print(report_mod.render(results, env, args.json or default_json(bench), getattr(bench, "EXTRA_COLUMNS", ())))

    return 1 if report_mod.failures(results) else 0


def bind(module):
    """The four entry points `test_ops_perf.py` and the CLI look for."""
    return (
        functools.partial(supported, module),
        functools.partial(default_json, module),
        functools.partial(run, module),
        functools.partial(main, module),
    )
