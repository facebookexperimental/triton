# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Gate and benchmark TorchTLX fusion cases against stock PT2.

All GPU attention cases: shared SDPA/flex helper; baseline SDPA, test flex/TLX "allow".
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import pathlib
import statistics
import sys
import types

import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
REL_L2_TOL = 1e-2


def load_runtime(case) -> None:
    global torch, triton
    import torch as torch_module
    import triton as triton_module  # @manual=//triton:triton
    import triton.language.extra.tlx.inductor.registry  # noqa: F401

    torch = torch_module
    triton = triton_module
    case.torch = torch_module


def attention(q, k, v, *, use_flex=False):
    return flex_attention(q, k, v) if use_flex else F.scaled_dot_product_attention(q, k, v)


def discover_cases() -> dict[str, object]:
    cases = {}
    for path in sorted(_HERE.glob("*_[0-9][0-9]_*.py")):
        case = importlib.import_module(f"torchtlx_benchmark.{path.stem}")
        if case.NAME != path.stem:
            raise ValueError(f"case name {case.NAME!r} does not match {path.name}")
        cases[case.NAME] = case
    return cases


def parse_args(cases):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=cases, default=next(iter(cases)))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--samples", type=int, default=5)
    selected, _ = parser.parse_known_args()
    cases[selected.case].add_arguments(parser)
    args = parser.parse_args()
    if min(args.samples, args.warmup, args.rep) <= 0:
        parser.error("--samples, --warmup and --rep must be positive")
    return args


def selected_shapes(case, args) -> tuple[dict[str, object], ...]:
    shapes = tuple(dict(shape) for shape in getattr(case, "SHAPES", ({}, )))
    for key in {key for shape in shapes for key in shape}:
        override = getattr(args, key, None)
        if override is None:
            continue
        matching = [shape for shape in shapes if shape.get(key) == override]
        if matching:
            shapes = tuple(matching)
        else:
            for shape in shapes:
                shape[key] = override
    return shapes


def flatten(output):
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        return [t for item in output for t in flatten(item)]
    return []


def relative_l2(output, reference) -> float:
    worst = 0.0
    for out, ref in zip(flatten(output), flatten(reference), strict=True):
        if out.shape != ref.shape:
            raise AssertionError(f"shape {out.shape} != reference {ref.shape}")
        out, ref = out.double(), ref.double()
        if not bool(torch.isfinite(out).all() & torch.isfinite(ref).all()):
            return float("inf")
        ref_sq = float(ref.square().sum())
        err_sq = float((out - ref).square().sum())
        error = (err_sq / ref_sq)**0.5 if ref_sq else err_sq**0.5
        worst = max(worst, error)
    return worst


def gate(case, label, output, reference) -> None:
    error = relative_l2(output, reference)
    tol = getattr(case, "REL_L2_TOL", REL_L2_TOL)
    ok = error <= tol
    print(f"GATE {'PASS' if ok else 'FAIL'} {label}: rel_l2={error:.3e} tol={tol:.0e}")
    if not ok:
        raise AssertionError(f"{label}: relative-L2 gate failed")


def compile_variant(case, inputs, variant):
    config = case.BASELINE_CONFIG if variant == "baseline" else case.CANDIDATE_CONFIG
    compile_context = getattr(case, "compile_context", contextlib.nullcontext)
    models = getattr(case, "comparison_models", lambda: (case.model, case.model))()
    model = models[variant == "candidate"]
    with torch._inductor.config.patch(config), compile_context(variant):
        if isinstance(model, types.FunctionType):
            original = model
            model = types.FunctionType(original.__code__.replace(co_name=variant), original.__globals__, variant,
                                       original.__defaults__, original.__closure__)
            model.__kwdefaults__ = original.__kwdefaults__
        compiled = torch.compile(model, fullgraph=True, options=config)
        output = compiled(*inputs)
    torch.cuda.synchronize()
    return compiled, output


def gradients(output, inputs, grad_inputs, grad_outputs):
    tensors = tuple(inputs[i] for i in grad_inputs)
    return torch.autograd.grad(tuple(flatten(output)), tensors, grad_outputs, retain_graph=True, allow_unused=True)


def bench_us(fn, args) -> float:
    return float(triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep, return_mode="median")) * 1000


def time_interleaved(case, args, functions, phase) -> None:
    samples = {variant: [] for variant in functions}
    for sample in range(args.samples):
        order = ("baseline", "candidate") if sample % 2 == 0 else ("candidate", "baseline")
        measured = {}
        for variant in order:
            measured[variant] = bench_us(functions[variant], args)
            samples[variant].append(measured[variant])
        print(f"{phase} sample={sample + 1} baseline_us={measured['baseline']:.3f} "
              f"candidate_us={measured['candidate']:.3f}")
    baseline = statistics.median(samples["baseline"])
    candidate = statistics.median(samples["candidate"])
    print(f"FINAL {phase} {case.problem()} baseline_us={baseline:.3f} "
          f"candidate_us={candidate:.3f} speedup={baseline / candidate:.3f}x")


def run_case(case, args) -> None:
    print(f"case={case.NAME} {case.problem()} device={torch.cuda.get_device_name(0)}")
    torch.manual_seed(0)
    inputs = case.make_inputs()
    grad_inputs = getattr(case, "GRAD_INPUTS", ())
    for i in grad_inputs:
        inputs[i].requires_grad_(True)
    torch._dynamo.reset()
    compiled, outputs = {}, {}
    for variant in ("baseline", "candidate"):
        compiled[variant], outputs[variant] = compile_variant(case, inputs, variant)
    gate(case, "forward candidate vs baseline", outputs["candidate"], outputs["baseline"])
    if grad_inputs:
        run_backward(case, args, inputs, grad_inputs, outputs)
    del outputs
    functions = {variant: (lambda fn=fn: fn(*inputs)) for variant, fn in compiled.items()}
    time_interleaved(case, args, functions, "forward")


def run_backward(case, args, inputs, grad_inputs, outputs) -> None:
    torch.manual_seed(1)
    grad_outputs = tuple(torch.randn_like(t) / max(1, t.shape[-1])**0.5 for t in flatten(outputs["baseline"]))
    functions = {
        variant: (lambda out=out: gradients(out, inputs, grad_inputs, grad_outputs))
        for variant, out in outputs.items()
    }
    grads = {variant: fn() for variant, fn in functions.items()}
    gate(case, "gradients candidate vs baseline", grads["candidate"], grads["baseline"])
    eager_grads = gradients(case.model(*inputs), inputs, grad_inputs, grad_outputs)
    for variant in grads:
        print(f"INFO {variant} gradients vs eager: rel_l2={relative_l2(grads[variant], eager_grads):.3e}")
    del grads, eager_grads
    time_interleaved(case, args, functions, "backward")


def main() -> None:
    cases = discover_cases()
    args = parse_args(cases)
    if args.list:
        for case in cases.values():
            print(f"{case.NAME}: {getattr(case, 'SHAPES', ())}")
        return
    case = cases[args.case]
    load_runtime(case)
    for shape in selected_shapes(case, args):
        for name, value in shape.items():
            setattr(args, name, value)
        case.configure(args)
        run_case(case, args)


if __name__ == "__main__":
    main()
