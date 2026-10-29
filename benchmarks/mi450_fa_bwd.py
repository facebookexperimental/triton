"""Correctness, tuning, and matched aiter comparison for gfx1250 backward.

Run GPU modes under gpu-lock. ``compile`` uses an explicit target and no GPU.
Results are appended as JSONL; ``--artifacts`` saves the generated assembly.
"""
import argparse
from collections import Counter, defaultdict
import ctypes
import dataclasses
import itertools
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import (
    _compiler_options,
    _delayed_bwd_kernel,
    _dq_slices,
    _tail_partition,
    allocate_workspace,
    fa_backward,
    select_config,
)


@triton.jit
def _forward_state(Q, K, V, O, LSE, HQ: tl.constexpr, HK: tl.constexpr, N: tl.constexpr, CAUSAL: tl.constexpr = False):
    head = tl.program_id(1)
    kv_head = head // HQ * HK + head % HQ // (HQ // HK)
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    cols = tl.arange(0, 128)
    q = tl.load(Q + head.to(tl.int64) * N * 128 + rows[:, None] * 128 + cols[None, :])
    acc = tl.zeros((64, 128), tl.float32)
    maximum = tl.full((64, ), float("-inf"), tl.float32)
    total = tl.zeros((64, ), tl.float32)
    for start in range(0, N, 64):
        keys = start + tl.arange(0, 64)
        offsets = kv_head.to(tl.int64) * N * 128 + keys[:, None] * 128 + cols[None, :]
        k = tl.load(K + offsets)
        v = tl.load(V + offsets)
        scores = tl.dot(q, tl.trans(k)) * (128**-0.5 * 1.4426950408889634)
        if CAUSAL:
            # Key 0 is always visible, so every row's running maximum is finite.
            scores = tl.where(keys[None, :] <= rows[:, None], scores, float("-inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(maximum - new_max)
        p = tl.exp2(scores - new_max[:, None])
        total = total * alpha + tl.sum(p, 1)
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        maximum = new_max
    tl.store(O + head.to(tl.int64) * N * 128 + rows[:, None] * 128 + cols[None, :], acc / total[:, None])
    tl.store(LSE + head.to(tl.int64) * N + rows, (maximum + tl.log2(total)) * 0.6931471805599453)


def make_case(shape, reference=False, seed=0, causal=False):
    b, hq, hk, n = shape
    torch.manual_seed(seed)
    q, do = [torch.randn((b, hq, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    k, v = [torch.randn((b, hk, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    o = torch.empty_like(q)
    lse = torch.empty((b, hq, n), device="cuda", dtype=torch.float32)
    if not reference:
        _forward_state[(n // 64, b * hq)](q, k, v, o, lse, hq, hk, n, causal, num_warps=4)
        return (q, k, v, o, do, lse), None
    dq = torch.empty_like(q, dtype=torch.float32)
    dk, dv = [torch.zeros_like(k, dtype=torch.float32) for _ in range(2)]
    for batch in range(b):
        for head in range(hq):
            kh = head // (hq // hk)
            qr = q[batch, head].float().requires_grad_()
            kr = k[batch, kh].float().requires_grad_()
            vr = v[batch, kh].float().requires_grad_()
            scores = qr @ kr.T * 128**-0.5
            if causal:
                scores = scores.masked_fill(~torch.ones_like(scores, dtype=torch.bool).tril(), float("-inf"))
            probs = scores.softmax(-1)
            out = probs @ vr
            grads = torch.autograd.grad(out, (qr, kr, vr), do[batch, head].float())
            o[batch, head] = out.detach()
            lse[batch, head] = scores.detach().logsumexp(-1)
            dq[batch, head] = grads[0]
            dk[batch, kh] += grads[1]
            dv[batch, kh] += grads[2]
    return (q, k, v, o, do, lse), (dq, dk, dv)


def errors(actual, expected):
    result = {}
    for name, a, e in zip(("dq", "dk", "dv"), actual, expected):
        diff = a.float() - e.float()
        nrmse = (diff.square().mean() / e.float().square().mean()).sqrt().item()
        max_abs = diff.abs().max().item()
        result[name] = {"nrmse": nrmse, "max_abs": max_abs}
        if not torch.isfinite(a).all().item() or not nrmse < 0.012:
            raise AssertionError(f"{name}: {result[name]}")
    return result


def fp32_backward_reference(case, *, query_block=256, head_block=4, causal=False):
    """Independent chunked PyTorch reference, with full FP32 softmax/gradients.

    Recompute probabilities from Q/K rather than using the providers' forward
    state. Query chunks bound memory at long lengths. TF32 is disabled so this
    also checks the large cases without treating AITER as a numeric oracle.
    """
    q, k, v, _, do, _ = case
    b, hq, n, d = q.shape
    hk = k.shape[1]
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for batch in range(b):
            for first in range(0, hq, head_block):
                last = min(first + head_block, hq)
                heads = torch.arange(first, last, device=q.device) // (hq // hk)
                keys = k[batch, heads].float()
                values = v[batch, heads].float()
                dkeys = torch.zeros_like(keys)
                dvalues = torch.zeros_like(values)
                for start in range(0, n, query_block):
                    end = min(start + query_block, n)
                    queries = q[batch, first:last, start:end].float()
                    dout = do[batch, first:last, start:end].float()
                    scores = torch.bmm(queries, keys.transpose(1, 2)) * d**-0.5
                    if causal:
                        hidden = torch.arange(n, device=q.device)[None, :] > torch.arange(start, end,
                                                                                          device=q.device)[:, None]
                        scores = scores.masked_fill(hidden, float("-inf"))
                    probs = scores.softmax(-1)
                    dp = torch.bmm(dout, values.transpose(1, 2))
                    ds = probs * (dp - (probs * dp).sum(-1, keepdim=True)) * d**-0.5
                    dq[batch, first:last, start:end] = torch.bmm(ds, keys)
                    dkeys += torch.bmm(ds.transpose(1, 2), queries)
                    dvalues += torch.bmm(probs.transpose(1, 2), dout)
                for offset, head in enumerate(range(first, last)):
                    dk[batch, head // (hq // hk)] += dkeys[offset]
                    dv[batch, head // (hq // hk)] += dvalues[offset]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return dq, dk, dv


class AiterBackward:
    """Call the default gfx1250 ASM kernels, including fresh dQ zeroing."""

    def __init__(self, case, library, split_batches=True, causal=False):
        self.case = case
        self.causal = causal
        q, k, v, o, do, lse = case
        self.b, self.hq, self.n, _ = q.shape
        self.hk = k.shape[1]
        # The default gfx1250 binary miscomputes dQ beyond 4 GiB across
        # batches. Rebase independent batch chunks; retain an unsplit mode
        # for reproducing the failure under the correctness guard.
        per_batch_bytes = self.hq * self.n * 128 * 4
        max_batch = max(1, (1 << 32) // per_batch_bytes) if split_batches else self.b
        self.batch_chunks = [min(max_batch, self.b - start) for start in range(0, self.b, max_batch)]
        self.delta = torch.empty_like(lse)
        self.dqa = torch.empty_like(q, dtype=torch.float32)
        self.dq = torch.empty_like(q)
        self.dk_exp = torch.empty_like(q)
        self.dv_exp = torch.empty_like(q)
        self.dk, self.dv = torch.empty_like(k), torch.empty_like(v)
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.fn = self.lib.mi450_aiter_bwd
        self.fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_void_p]
        self.fn.restype = ctypes.c_float

    def __call__(self):
        q, k, v, o, do, lse = self.case
        tensors = (q, k, v, o, do, lse, self.delta, self.dqa, self.dq, self.dk_exp, self.dv_exp)
        start = 0
        for chunk in self.batch_chunks:
            status = self.fn(*(t[start:].data_ptr() for t in tensors), chunk, self.hq, self.hk, self.n,
                             int(self.causal), 128**-0.5,
                             torch.cuda.current_stream().cuda_stream)
            if status < 0:
                raise RuntimeError(f"aiter did not select a kernel: {status}")
            start += chunk
        if self.hq != self.hk:
            shape = (self.b, self.hk, self.hq // self.hk, self.n, 128)
            torch.sum(self.dk_exp.view(shape), dim=2, out=self.dk)
            torch.sum(self.dv_exp.view(shape), dim=2, out=self.dv)
        else:
            self.dk, self.dv = self.dk_exp, self.dv_exp
        return self.dq, self.dk, self.dv


def inspect_assembly(assembly):
    """Count the steady loop using backward branches, retaining source attribution.

    Counts are static ISA instructions per loop body and per wave, not profiler
    cycles. Select the WMMA-containing backward region with the most WMMAs;
    when several branches close it, retain the complete region.
    """
    lines = assembly.splitlines()
    labels = {}
    branches = []
    for index, line in enumerate(lines):
        label = re.match(r"^([.\w]+):", line)
        if label:
            labels[label[1]] = index
        branch = re.match(r"\s*s_(?:cbranch_\w+|branch)\s+([.\w]+)", line)
        if branch and branch[1] in labels and labels[branch[1]] < index:
            start = labels[branch[1]]
            count = sum("v_wmma_" in item for item in lines[start:index + 1])
            if count:
                branches.append((count, index - start, start, index + 1))
    if not branches:
        return {"loop_found": False}
    _, _, start, end = max(branches)
    counts = Counter()
    lds_sources = defaultdict(Counter)
    atomic_addresses = defaultdict(set)
    atomic_offsets = defaultdict(list)
    atomic_wmma_positions = defaultdict(list)
    wmma_position = 0
    atomic_clauses = []
    clause_remaining = 0
    # Inherit the debug location immediately preceding the loop.
    location = "unknown"
    for index, line in enumerate(lines[:end]):
        if line.lstrip().startswith(".loc"):
            locations = re.findall(r"([\w.]+\.py):(\d+):(\d+)", line)
            if locations:
                filename, row, _ = locations[-1]
                location = f"{filename}:{row}"
        if index < start:
            continue
        instruction = re.match(r"\s+([a-z][\w.]*)\s", line)
        if not instruction:
            continue
        opcode = instruction[1]
        counts[opcode] += 1
        if opcode.startswith("v_wmma_"):
            wmma_position += 1
        if opcode == "s_clause":
            clause_remaining = int(line.split()[1], 0) + 1
            atomic_clauses.append(0)
        elif clause_remaining:
            clause_remaining -= 1
            if opcode.startswith("buffer_atomic_"):
                atomic_clauses[-1] += 1
        if opcode.startswith("buffer_atomic_"):
            atomic_wmma_positions[opcode].append(wmma_position)
            address = re.search(r",\s*(v\d+)(?:\s*/\*(v\d+)\*/)?\s*,\s*s\[", line)
            if address:
                atomic_addresses[opcode].add(address[2] or address[1])
            immediate = re.search(r"\boffset:(\d+)", line)
            atomic_offsets[opcode].append(int(immediate[1]) if immediate else 0)
        if opcode.startswith("ds_load"):
            lds_sources[location][opcode] += 1
    return {
        "loop_found": True,
        "loop_start_line": start + 1,
        "loop_end_line": end,
        "loop_instruction_counts": dict(sorted(counts.items())),
        "loop_atomic_address_registers": {key: sorted(value)
                                          for key, value in sorted(atomic_addresses.items())},
        "loop_atomic_immediate_offsets_bytes": dict(sorted(atomic_offsets.items())),
        "loop_atomic_wmma_positions": dict(sorted(atomic_wmma_positions.items())),
        "loop_atomic_clause_sizes": [size for size in atomic_clauses if size],
        "loop_lds_loads": sum(value for key, value in counts.items() if key.startswith("ds_load")),
        "loop_lds_loads_by_source": {key: dict(value)
                                     for key, value in sorted(lds_sources.items())},
        "loop_assembly": "\n".join(lines[start:end]) + "\n",
    }


def compile_kernel(config, artifacts, shape, *, compile_options=None, constants_overrides=None, kernel_fn=None,
                   signature_overrides=None, compiler_options_fn=None):
    kernel_fn = _delayed_bwd_kernel if kernel_fn is None else kernel_fn
    compiler_options_fn = (
        lambda c: _compiler_options(c, shape[3])) if compiler_options_fn is None else compiler_options_fn
    constants = dict(HQ=shape[1], HK=shape[2], N=shape[3], SM_SCALE=128**-0.5, BM=config.block_m, BN=config.block_n,
                     NW=config.num_warps, TDM=config.use_tdm, STAGES=config.stages, DELAYED=config.delayed,
                     UNROLL=config.unroll, FRAGMENT_FENCES=config.fragment_fences,
                     FRAGMENT_FENCE_MASK=config.fragment_fence_mask, FRAGMENTED=config.fragmented,
                     RESIDENT_KV=config.resident_kv, ORDERED=config.ordered, PACKED_DQ=config.packed_dq,
                     REMATERIALIZE=config.rematerialize, OUTER_M=config.outer_m, ASYNC_STATS=config.async_stats,
                     Q_SPLITS=config.q_splits, REUSE_Q=config.reuse_q, PINGPONG=config.pingpong,
                     SPREAD_DQ=config.spread_dq, LOG2_LSE=config.log2_lse, CACHE_SCORE_K=config.cache_score_k)
    optional_constants = {
        "TILE_MAJOR": ("tile_major", False),
        "CLUSTER_DQ": ("cluster_dq", False),
        "PARTITION_LDS": ("partition_lds", False),
        "FRONT_REGIONS": ("front_regions", 0),
        "BRIDGE_REGIONS": ("bridge_regions", 0),
    }
    for name, (field, default) in optional_constants.items():
        if name in kernel_fn.arg_names:
            constants[name] = getattr(config, field, default)
    if "DQ_SLICES" in kernel_fn.arg_names:
        constants["DQ_SLICES"] = max(1, shape[3] // config.block_n // 16) if config.cluster_dq else 1
    for name, field in (("OCC1", "occ1"), ("OCC1_K_BANDS", "occ1_k_bands"), ("OCC1_PREFETCH", "occ1_prefetch"),
                        ("OCC1_SCHED", "occ1_sched"), ("PHASE_IGLP", "phase_iglp"), ("OCC1_DQ_FIRST", "occ1_dq_first"),
                        ("OCC1_LOAD_AFTER_DK", "occ1_load_after_dk")):
        if name in kernel_fn.arg_names:
            constants[name] = getattr(config, field)
    if "CAUSAL" in kernel_fn.arg_names:
        constants["CAUSAL"] = False
    # Mirror the fa_backward launch: split dK/dV stores and the tail-split
    # partition change the epilogue and the pointer signature.
    b, hq, hk, n = shape
    tail_splits = getattr(config, "tail_splits", 1)
    tail_base, tail_owners = _tail_partition(config, b, hk, n)
    if "SPLIT_KV_STORES" in kernel_fn.arg_names:
        constants["SPLIT_KV_STORES"] = config.occ1 and config.packed_dq and config.q_splits > 1
    if "TAIL_SPLITS" in kernel_fn.arg_names:
        constants.update(TAIL_SPLITS=tail_splits, TAIL_BASE=tail_base, TAIL_OWNERS=tail_owners)
        if tail_splits == 1:
            constants.update(TAIL_DK=None, TAIL_DV=None)
    if constants_overrides:
        constants.update(constants_overrides)
    signature = {name: "*bf16" for name in ("Q", "K", "V", "DO", "DK", "DV")}
    signature.update({name: "*fp32" for name in ("LSE", "Delta")})
    signature["DQA"] = "*bf16" if config.packed_dq else "*fp32"
    if config.q_splits > 1 or tail_splits > 1:
        signature.update(DK="*fp32", DV="*fp32")
    if tail_splits > 1:
        signature.update(TAIL_DK="*bf16", TAIL_DV="*bf16")
    signature.update(signature_overrides or {})
    # Match the pointer specialization of the fresh Torch allocations used
    # by make_case/allocate_workspace, including vector buffer-copy controls
    # and the 32-bit pointer range of buffers below 2 GiB.
    attrs = {(kernel_fn.arg_names.index(name), ): [("tt.divisibility", 16)] for name in signature}
    qe, ke = b * hq * n * 128, b * hk * n * 128
    if tail_splits > 1:
        partial_bytes = tail_splits * tail_owners * config.block_n * 128 * 4
    else:
        partial_bytes = ke * 4 * (config.q_splits if config.occ1 and config.packed_dq else 1)
    split = config.q_splits > 1 or tail_splits > 1
    sizes = dict(Q=qe * 2, K=ke * 2, V=ke * 2, DO=qe * 2, LSE=b * hq * n * 4, Delta=b * hq * n * 4,
                 DQA=qe * (2 if config.packed_dq else 4) * _dq_slices(config, n), DK=partial_bytes if split else ke * 2,
                 DV=partial_bytes if split else ke * 2, TAIL_DK=ke * 2, TAIL_DV=ke * 2)
    for name in signature:
        if sizes.get(name, 2**31) <= 2**31 - 1:
            attrs[(kernel_fn.arg_names.index(name), )].append(("tt.pointer_range", 32))
    kernel = triton.compile(ASTSource(kernel_fn, signature, constants,
                                      attrs=attrs), target=GPUTarget("hip", "gfx1250", 32),
                            options={**compiler_options_fn(config), **(compile_options or {})})
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
        config_hash = hashlib.sha256(json.dumps(dataclasses.asdict(config), sort_keys=True).encode()).hexdigest()[:16]
        # Prototype constexpr overrides can share a BackwardConfig. Include
        # the compiled hash so their assembly and IR cannot overwrite each other.
        tag = "_".join(map(str, shape)) + "_" + config_hash + "_" + kernel.hash[:16]
        (artifacts / f"{tag}.config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")
        (artifacts / f"{tag}.compile.json").write_text(
            json.dumps(
                {
                    "compiled_kernel_hash": kernel.hash,
                    "constants_overrides": constants_overrides or {},
                    "signature_overrides": signature_overrides or {},
                    "compile_options": compile_options or {},
                }, indent=2) + "\n")
        for extension in ("ttgir", "llir", "amdgcn"):
            (artifacts / f"{tag}.{extension}").write_text(kernel.asm[extension])
    assembly = kernel.asm["amdgcn"]
    metadata = {
        "compiled_kernel_hash": kernel.hash,
        "shared": kernel.metadata.shared,
        "wmma": assembly.count("v_wmma_"),
        "tdm": assembly.count("tensor_load_to_lds"),
        "pointer_alignment": 16,
    }
    for field in ("vgpr_count", "sgpr_count", "private_segment_fixed_size"):
        metadata[field] = int(re.search(r"\." + field + r":\s*(\d+)", assembly).group(1))
    analysis = inspect_assembly(assembly)
    loop_assembly = analysis.pop("loop_assembly", "")
    if artifacts and loop_assembly:
        (artifacts / f"{tag}.loop.s").write_text(loop_assembly)
    metadata.update(analysis)
    if compile_options:
        metadata["compile_options"] = compile_options
    return metadata


def benchmark(fn, repeats):
    import triton.testing
    return [triton.testing.do_bench_cudagraph(fn, rep=repeats, return_mode="median") for _ in range(3)]


def sustained_benchmark(fn, estimate_ms, seconds=10.0):
    """Replay complete calls continuously while sampling this GPU's actual limits.

    A roughly 250 ms graph batch bounds host gaps. CUDA events measure batches;
    telemetry is read-only and runs on a separate CPU thread. This complements
    the short paired protocol; it does not lock clocks or change power limits.
    """
    if not os.environ.get("GPU_LOCK_HELD"):
        raise RuntimeError("sustained GPU measurement requires gpu-lock")
    if seconds <= 0 or estimate_ms <= 0:
        raise ValueError("duration and latency estimate must be positive")
    static_command = ["amd-smi", "static", "-g", "0", "--asic", "--limit", "--partition", "--json"]
    metric_command = ["amd-smi", "metric", "-g", "0", "--clock", "--power", "--usage", "--json"]
    static = json.loads(subprocess.check_output(static_command, text=True, timeout=10))
    batch_calls = max(1, int(250 / estimate_ms) + 1)
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(batch_calls):
            fn()
    warmup_end = time.monotonic() + 2.0
    while time.monotonic() < warmup_end:
        graph.replay()
        torch.cuda.synchronize()
    stop = threading.Event()
    telemetry = []

    def value(data):
        return data.get("value") if isinstance(data, dict) else None

    def sample():
        while not stop.is_set():
            now = time.time()
            try:
                data = json.loads(subprocess.check_output(metric_command, text=True, timeout=5))["gpu_data"][0]
                telemetry.append({
                    "time": now,
                    "socket_power_w": value(data["power"].get("socket_power")),
                    "gfx_clock_mhz": {
                        key: value(clock.get("clk"))
                        for key, clock in data["clock"].items()
                        if key.startswith("gfx_") and isinstance(clock, dict)
                    },
                    "gfx_activity_pct": value(data["usage"].get("gfx_activity")),
                })
            except (subprocess.SubprocessError, json.JSONDecodeError, KeyError) as exc:
                telemetry.append({"time": now, "error": str(exc)})
            stop.wait(1.0)

    monitor = threading.Thread(target=sample, daemon=True)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    wall_start = time.time()
    clock_start = time.monotonic()
    monitor.start()
    try:
        while time.monotonic() - clock_start < seconds:
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / batch_calls)
    finally:
        wall_seconds = time.monotonic() - clock_start
        stop.set()
        monitor.join(timeout=6)
    if monitor.is_alive():
        raise RuntimeError("telemetry worker did not stop")
    return dict(ms=statistics.median(samples), mean_ms=statistics.mean(samples), samples_ms=samples,
                calls=len(samples) * batch_calls, calls_per_batch=batch_calls, wall_start=wall_start,
                wall_seconds=wall_seconds, telemetry=telemetry, smi_static=static,
                telemetry_commands=[static_command, metric_command], warmup_seconds=2.0)


def measurement_identity(boot_id_path=Path("/proc/sys/kernel/random/boot_id")):
    """Identify the host boot so paired results cannot silently span recovery."""
    boot_id = boot_id_path.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot_id):
        raise ValueError("measurement requires a valid Linux boot ID")
    return {"boot_id": boot_id, "hostname": os.uname().nodename}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compile", "check", "tune", "pipeline", "bench"), default="check")
    parser.add_argument("--shape", default="1,4,2,512", help="B,Hq,Hkv,N")
    parser.add_argument("--config", help="JSON BackwardConfig fields")
    parser.add_argument("--output", type=Path, default=Path("mi450-bwd-results.jsonl"))
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--aiter-library", type=Path)
    parser.add_argument("--aiter-unsplit", action="store_true", help="Reproduce the raw aiter batch-addressing failure")
    parser.add_argument("--fp32-reference", action="store_true",
                        help="Check bench outputs against chunked FP32 PyTorch")
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument("--sustained-seconds", type=float, default=0,
                        help="Additional continuous bench with clock/power telemetry; 0 disables")
    parser.add_argument("--num-cus", type=int, default=192, help="CU count for offline compile policy selection")
    parser.add_argument("--causal", action="store_true",
                        help="Causal mask; TFLOPS then count half of the 10·B·Hq·N²·D non-causal FLOPs")
    args = parser.parse_args()
    if args.sustained_seconds < 0 or args.sustained_seconds and args.mode != "bench":
        parser.error("--sustained-seconds requires bench mode and a non-negative duration")
    shape = tuple(map(int, args.shape.split(",")))
    overrides = json.loads(args.config) if args.config else {}
    config = select_config(shape[-1], *shape[:3], num_cus=args.num_cus if args.mode == "compile" else None,
                           packed_dq=overrides.get("packed_dq", True), causal=args.causal)
    config = dataclasses.replace(config, **overrides)
    kernel_sha256 = hashlib.sha256(
        Path(__file__).parents[1].joinpath(
            "third_party/tlx/ops/kernels/flash_attn/gfx1250_bwd.py").read_bytes()).hexdigest()
    identity = measurement_identity()
    flops = (5 if args.causal else 10) * shape[0] * shape[1] * shape[3]**2 * 128

    def emit(record):
        record["time"] = time.time()
        record["causal"] = args.causal
        record["kernel_sha256"] = kernel_sha256
        record["experiment"] = os.environ.get("MI450_EXPERIMENT", "initial")
        record.update(identity)
        print(json.dumps(record), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as f:
            f.write(json.dumps(record) + "\n")

    if args.mode == "compile":
        emit({
            "mode":
            "compile", "shape":
            shape, "config":
            dataclasses.asdict(config), **compile_kernel(
                config, args.artifacts, shape, constants_overrides=dict(CAUSAL=True, DQ_SLICES=_dq_slices(
                    config, shape[3], True)) if args.causal else None, compiler_options_fn=lambda c: _compiler_options(
                    c, shape[3], args.causal))
        })
        return
    if not os.environ.get("GPU_LOCK_HELD"):
        raise RuntimeError("Run GPU modes under gpu-lock and pass GPU_LOCK_HELD through Docker")
    case, reference = make_case(shape, reference=args.mode == "check", causal=args.causal)
    if args.fp32_reference and args.mode == "bench":
        reference = fp32_backward_reference(case, causal=args.causal)
    workspace = allocate_workspace(case[0], case[1], config=config, causal=args.causal)
    configs = [config]
    if args.mode in ("tune", "pipeline"):
        if args.mode == "tune":
            configs = [
                dataclasses.replace(config, block_m=bm, block_n=bn, num_warps=nw, use_tdm=tdm,
                                    async_stats=config.async_stats and tdm)
                for bm, bn, nw, tdm in itertools.product((32, 64), (128, 256), (2, 4, 8), (False, True))
            ]
        else:
            configs = [
                dataclasses.replace(config, stages=stages, unroll=unroll)
                for stages, unroll in itertools.product((2, 3, 4), (1, 2, 4, 8))
            ]
        check_case, check_reference = make_case((1, 4, 2, 512), reference=True, causal=args.causal)
        check_workspace = allocate_workspace(check_case[0], check_case[1], config=config, causal=args.causal)
    for config in configs:
        record = {"mode": args.mode, "shape": shape, "provider": "tlx", "config": dataclasses.asdict(config)}
        try:
            if args.mode == "check":
                record["errors"] = errors(fa_backward(*case, causal=args.causal, config=config, workspace=workspace),
                                          reference)
                # A second invocation catches missing per-call accumulator clearing.
                record["repeat_errors"] = errors(
                    fa_backward(*case, causal=args.causal, config=config, workspace=workspace), reference)
            else:
                if args.fp32_reference and args.mode == "bench":
                    record["errors"] = errors(
                        fa_backward(*case, causal=args.causal, config=config, workspace=workspace), reference)
                    record["repeat_errors"] = errors(
                        fa_backward(*case, causal=args.causal, config=config, workspace=workspace), reference)
                if args.mode in ("tune", "pipeline"):
                    record["errors"] = errors(
                        fa_backward(*check_case, causal=args.causal, config=config, workspace=check_workspace),
                        check_reference)
                samples = benchmark(lambda: fa_backward(*case, causal=args.causal, config=config, workspace=workspace),
                                    args.repeat_ms)
                record.update(ms=statistics.median(samples), samples_ms=samples,
                              tflops=flops / statistics.median(samples) / 1e9)
                if args.sustained_seconds:
                    record["sustained"] = sustained_benchmark(
                        lambda: fa_backward(*case, causal=args.causal, config=config, workspace=workspace),
                        record["ms"], args.sustained_seconds)
            emit(record)
        except Exception as exc:
            if args.mode not in ("tune", "pipeline"):
                raise
            emit({**record, "error": str(exc)})
    if args.aiter_library:
        aiter = AiterBackward(case, args.aiter_library, split_batches=not args.aiter_unsplit, causal=args.causal)
        record = {"mode": args.mode, "shape": shape, "provider": "aiter", "batch_chunks": aiter.batch_chunks}
        if args.mode == "check":
            record["errors"] = errors(aiter(), reference)
            record["repeat_errors"] = errors(aiter(), reference)
        else:
            if args.mode == "bench":
                record["tlx_vs_aiter_errors"] = errors(tuple(workspace[2:5]), aiter())
                if args.fp32_reference:
                    record["errors"] = errors(aiter(), reference)
            samples = benchmark(aiter, args.repeat_ms)
            record.update(ms=statistics.median(samples), samples_ms=samples,
                          tflops=flops / statistics.median(samples) / 1e9)
            if args.sustained_seconds:
                record["sustained"] = sustained_benchmark(aiter, record["ms"], args.sustained_seconds)
        emit(record)


if __name__ == "__main__":
    main()
