"""Production gfx942 addmm adapter for the shared GEMM tuning harness.

Reuse the MM harness's full-space oracle, profiling, and policy evaluation.
Candidate calls go through tlx.ops.addmm with the recorded bias contract.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import functools
import sys
from pathlib import Path

import torch

# Load a private copy so adapting its globals cannot affect an MM evaluation
# in the same process. Its relative profiler imports retain the package context.
_name = "third_party.tlx.tools.agents.kernel_optimization.decision_maker.tuning_harnesses._addmm_base"
_spec = importlib.util.spec_from_file_location(_name, Path(__file__).with_name("mm.py"))
_base = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _base
_spec.loader.exec_module(_base)
_mm_inputs = _base._inputs
_mm_workload_command = _base._profile_workload_command
_mm_record_tuners = _base._record_tuners
_mm_oracle_winner = _base._oracle_winner
_bias_inputs = {}
_outputs = {}
_measurement_counts = []
_split_k_recorders = []


def _measure(fn, repetitions):
    repository = str(_base._REPO_ROOT)
    if repository not in sys.path:
        sys.path.insert(0, repository)
    from python.test.tlx_benchmark._harness.measure import _gpu_event_samples, reject_outliers_iqr

    # Use the production kernel verdict metric: every call is queued before
    # its start event, excluding host dispatch gaps. Match its IQR policy.
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    samples = _gpu_event_samples(fn, max(500, int(repetitions)))
    kept = reject_outliers_iqr(samples)
    _measurement_counts.append({"raw": len(samples), "kept": len(kept)})
    return [sample * 1000.0 for sample in kept]


def _autotune_bench(fn, quantiles):
    samples = [sample / 1000.0 for sample in _measure(fn, 500)]
    from python.test.tlx_benchmark._harness.measure import percentiles

    return list(percentiles(samples, wanted=tuple(q * 100 for q in quantiles)))


def _key(a, b):
    return (a.shape, b.shape, a.stride(), b.stride(), a.dtype, a.device)


def _inputs(case, device):
    a, b = _mm_inputs(case, device)
    parameters = case["parameters"]
    m, n = a.shape[0], b.shape[1]
    strides = tuple(parameters.get("bias_strides", (1, )))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(parameters.get("seed", 0)) + 1)
    if len(strides) == 1:
        bias = _base._strided_random((n, ), strides, a.dtype, device, generator)
    else:
        bias = _base._strided_random((m, n), strides, a.dtype, device, generator)
    _bias_inputs[_key(a, b)] = bias
    return a, b


class _SplitKAutotuner:
    """Measure the complete existing split-K GEMM + reducer launch pair."""

    def __init__(self, module, shape):
        self.module = module
        self.shape = shape
        self.configs = module._candidate_configs(shape, enable_global_split_k=True)
        self.configs_timings = {}
        self.best_config = None
        self.cache = {}
        self.workspaces = {}

    def run(self, a, b, bias, out):
        key = (_key(a, b), bias.stride() if bias is not None else None)
        planes = max(c.kwargs.get("SPLIT_K", 1) for c in self.configs)
        workspace_key = (key, planes)
        if workspace_key not in self.workspaces:
            self.workspaces[workspace_key] = torch.empty((planes * a.shape[0], b.shape[1]), device=a.device,
                                                         dtype=torch.float32)
        workspace = self.workspaces[workspace_key]

        def launch(config):
            return self.module._launch_global_split_k_config(a, b, bias, out, workspace, config)

        if len(self.configs) == 1:
            selected = self.configs[0]
        elif key in self.cache:
            selected = self.cache[key]
        else:
            import triton
            for config in self.configs:
                try:
                    times = _autotune_bench(lambda config=config: launch(config), (0.5, 0.2, 0.8))
                except triton.OutOfResources:
                    times = [math.inf] * 3
                self.configs_timings[config] = times
            selected = min(self.configs_timings, key=lambda c: self.configs_timings[c][0])
            if not math.isfinite(self.configs_timings[selected][0]):
                raise RuntimeError("No valid full-space split-K addmm configuration")
            self.cache[key] = selected
        self.best_config = selected
        result = launch(selected)
        for recorder in _split_k_recorders:
            recorder["matmul_kernel_gfx942"] = self
        return result


def _install_split_k_adapter(module):
    original = getattr(module, "_launch_global_split_k", None)
    if not callable(original):
        return
    tuners = {}

    def tuner(shape):
        if shape not in tuners:
            tuners[shape] = _SplitKAutotuner(module, shape)
        return tuners[shape]

    def launch(a, b, bias, out, space):
        if space != "full":
            return original(a, b, bias, out, space)
        return tuner((a.shape[0], b.shape[1], a.shape[1])).run(a, b, bias, out)

    module._launch_global_split_k = launch
    module._tlx_agent_split_k_tuner = tuner


@contextlib.contextmanager
def _record_tuners(into):
    _split_k_recorders.append(into)
    try:
        with _mm_record_tuners(into):
            yield
    finally:
        _split_k_recorders.remove(into)


@contextlib.contextmanager
def _oracle_winner(artifact, a, b, winner):
    module = artifact.get("module")
    factory = getattr(module, "_tlx_agent_split_k_tuner", None)
    precheck = getattr(module, "_precheck_global_split_k", None)
    shape = (a.shape[0], b.shape[1], a.shape[1])
    key = _key(a, b)
    bias = _bias_inputs[key]
    if key not in _outputs:
        _outputs[key] = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    eligible = (factory is not None and callable(precheck) and module._global_split_k_shape(*shape)
                and precheck(a, b, bias, _outputs[key]))
    tuner = factory(shape) if factory is not None else None
    expected = winner.split(": ", 1)[-1]
    selected = [config for config in tuner.configs if _base._format_config(config) == expected] if tuner else []
    # An authoring candidate may add a split-K dispatch outside the incumbent
    # FP16 eligibility helper. Replay its recorded decomposition through the
    # public full-space call rather than trying the direct GEMM autotuner.
    recorded_split = len(selected) == 1 and selected[0].kwargs.get("SPLIT_K", 1) > 1
    if not eligible and not recorded_split:
        with _mm_oracle_winner(artifact, a, b, winner) as launch:
            yield launch
        return
    if len(selected) != 1:
        raise RuntimeError(f"could not resolve split-K addmm oracle winner {winner!r}")
    original = tuner.configs
    tuner.configs = selected
    try:
        yield lambda: artifact["op"](a, b, space="full")
    finally:
        tuner.configs = original


def _install_candidate(module, arch):
    import triton.tlx.ops as ops
    from triton.tlx.ops import _catalog

    candidate = getattr(module, "_gemm", None)
    if not callable(candidate) or arch != "gfx942":
        raise TypeError("addmm tuning requires the shared gfx942 _gemm implementation")
    _install_split_k_adapter(module)
    tuned = getattr(module, "_tuned", None)
    if callable(tuned):

        @functools.wraps(tuned)
        def queued_tuned(space, *args, **kwargs):
            tuner = tuned(space, *args, **kwargs)
            if space == "full":
                tuner._do_bench = _autotune_bench
            return tuner

        module._tuned = queued_tuned
    spec = _catalog._BY_KEY[("addmm", arch)]

    def implementation(bias, a, b, *, out=None, space="heuristic"):
        return candidate(a, b, bias, out=out, space=space)

    def call(a, b, *, space="heuristic"):
        key = _key(a, b)
        if key not in _outputs:
            _outputs[key] = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        original = ops.impl_for

        def impl_for(op, *args, **kwargs):
            return (implementation, spec) if op == "addmm" else original(op, *args, **kwargs)

        ops.impl_for = impl_for
        try:
            return ops.addmm(_bias_inputs[key], a, b, out=_outputs[key], space=space)
        finally:
            ops.impl_for = original

    return call


def _reference_addmm(a, b):
    return torch.addmm(_bias_inputs[_key(a, b)], a, b)


def _profile_workload_command(*args, **kwargs):
    command = _mm_workload_command(*args, **kwargs)
    command[1] = str(Path(__file__).resolve())
    return command


_base._inputs = _inputs
_base._install_candidate = _install_candidate
_base._reference_matmul = _reference_addmm
_base._profile_workload_command = _profile_workload_command
_base._measure = _measure
_base._record_tuners = _record_tuners
_base._oracle_winner = _oracle_winner
build = _base.build
verify = _base.verify
profile = _base.profile


def benchmark(artifact, case, repetitions):
    _measurement_counts.clear()
    result = _base.benchmark(artifact, case, repetitions)
    result["cache_policy"] = f"{artifact['phase']}_queued_gpu_events_iqr"
    result["metrics"].update({
        "latency_measure_mode": "gpu_events",
        "outlier_policy": "iqr",
        "measurement_sample_counts": list(_measurement_counts),
    })
    artifact["records"][case["case_id"]].update(result["metrics"])
    return result


if __name__ == "__main__":
    raise SystemExit(_base._run_profile_workload(_base._parse_profile_arguments()))
