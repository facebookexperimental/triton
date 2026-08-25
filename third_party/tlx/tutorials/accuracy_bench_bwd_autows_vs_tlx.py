"""Accuracy benchmark: AutoWS bwd vs TLX bwd vs torch reference.

The AutoWS config is selected with BENCH_BWD_IDX. Its NUM_CTAS and BLOCK_M1
select the matching handwritten TLX configuration.

Runs the backward several times on identical inputs to surface any
non-determinism (the TMEM-reuse race). Reports max-abs error vs torch
reference, AutoWS-vs-TLX max-abs diff, and run-to-run variation per impl.

Usage:
  TRITON_ALWAYS_COMPILE=1 python accuracy_bench_bwd_autows_vs_tlx.py
  (add TRITON_KERNEL_DUMP=1 to dump both kernels' final TTGIR to ~/.triton/dump)
"""
import os
import sys

import torch

TUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TUT)

import fused_attention_ws_device_tma as aws  # noqa: E402
import blackwell_fa_ws_pipelined_persistent as tlx  # noqa: E402

DEVICE = "cuda"
Z = int(os.environ.get("BENCH_Z", 8))
H = int(os.environ.get("BENCH_H", 16))
N_CTX = int(os.environ.get("BENCH_NCTX", 1024))
HEAD_DIM = int(os.environ.get("BENCH_D", 128))
CAUSAL = os.environ.get("BENCH_CAUSAL", "0") == "1"
SM_SCALE = 0.5
DTYPE = torch.float16
RUNS = int(os.environ.get("BENCH_RUNS", 5))
SEED = int(os.environ.get("BENCH_SEED", 20))

# baseVariant: "ws_persistent" (failing) or "ws" (passing). Default to the
# failing persistent path under investigation.
AWS_VARIANT = os.environ.get("BENCH_VARIANT", "ws_persistent")
BWD_IDX = int(os.environ.get("BENCH_BWD_IDX", 4))


def pin_autows_config():
    kern = aws._attn_bwd_persist if AWS_VARIANT == "ws_persistent" else aws._attn_bwd
    kern.configs = [aws.configs_bwd_persist[BWD_IDX]]
    kern.cache = {}


def pin_tlx_config():
    aws_cfg = aws.configs_bwd_persist[BWD_IDX]
    num_ctas = aws_cfg.kwargs.get("NUM_CTAS", 1)
    block_m = aws_cfg.kwargs["BLOCK_M1"]
    cfgs = [
        c for c in tlx.BWD_CONFIGS
        if c.kwargs.get("NUM_CTAS", 1) == num_ctas and c.kwargs.get("USE_WARP_BARRIER") is False and (
            num_ctas > 1 or c.kwargs.get("BLOCK_M1") == block_m)
    ]
    assert len(cfgs) == 1, f"expected 1 matching TLX config, got {len(cfgs)}"
    tlx._attn_bwd_ws.configs = cfgs
    tlx._attn_bwd_ws.cache = {}


def make_inputs(seed=20):
    torch.manual_seed(seed)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE).normal_(0.0, 0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE).normal_(0.0, 0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=DTYPE, device=DEVICE).normal_(0.0, 0.5)
    dout = torch.randn_like(q)
    return q, k, v, dout


def torch_ref(q, k, v, dout):
    q = q.clone().requires_grad_()
    k = k.clone().requires_grad_()
    v = v.clone().requires_grad_()
    p = torch.matmul(q, k.transpose(2, 3)) * SM_SCALE
    if CAUSAL:
        M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(DTYPE)
    out = torch.matmul(p, v).half()
    out.backward(dout)
    return out.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def run_autows(q, k, v, dout):
    q = q.clone().requires_grad_()
    k = k.clone().requires_grad_()
    v = v.clone().requires_grad_()
    out = aws.attention(q, k, v, CAUSAL, SM_SCALE, AWS_VARIANT, False, 0, False).half()
    out.backward(dout)
    return out.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def run_tlx(q, k, v, dout):
    q = q.clone().requires_grad_()
    k = k.clone().requires_grad_()
    v = v.clone().requires_grad_()
    out = tlx.attention(q, k, v, SM_SCALE, CAUSAL).half()
    out.backward(dout)
    return out.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    pin_autows_config()
    pin_tlx_config()
    q, k, v, dout = make_inputs(SEED)

    ro, rdq, rdk, rdv = torch_ref(q, k, v, dout)

    names = ["dq", "dk", "dv"]
    aws_runs, tlx_runs = [], []
    aws_fail = tlx_fail = None
    for i in range(RUNS):
        try:
            _, adq, adk, adv = run_autows(q, k, v, dout)
            aws_runs.append((adq, adk, adv))
        except Exception as e:  # noqa: BLE001
            aws_fail = f"{type(e).__name__}: {str(e)[:160]}"
        try:
            _, tdq, tdk, tdv = run_tlx(q, k, v, dout)
            tlx_runs.append((tdq, tdk, tdv))
        except Exception as e:  # noqa: BLE001
            tlx_fail = f"{type(e).__name__}: {str(e)[:160]}"
    if aws_fail:
        print(f"AutoWS FAILED to run: {aws_fail}")
    if tlx_fail:
        print(f"TLX FAILED to run: {tlx_fail}")
    if not aws_runs or not tlx_runs:
        print("Cannot compare: one implementation did not produce results.")
        return

    def vs_ref(runs, ref):
        return [max(maxabs(r[j], ref[j]) for r in runs) for j in range(3)]

    def run_to_run(runs):
        out = []
        for j in range(3):
            mx = 0.0
            for a in range(len(runs)):
                for b in range(a + 1, len(runs)):
                    mx = max(mx, maxabs(runs[a][j], runs[b][j]))
            out.append(mx)
        return out

    ref = (rdq, rdk, rdv)
    aws_err = vs_ref(aws_runs, ref)
    tlx_err = vs_ref(tlx_runs, ref)
    aws_var = run_to_run(aws_runs)
    tlx_var = run_to_run(tlx_runs)
    cross = [maxabs(aws_runs[0][j], tlx_runs[0][j]) for j in range(3)]

    print(f"\nConfig: Z={Z} H={H} N_CTX={N_CTX} d={HEAD_DIM} causal={CAUSAL} "
          f"sm_scale={SM_SCALE} runs={RUNS}")
    print(f"{'tensor':6} {'AutoWS vs ref':>14} {'TLX vs ref':>12} "
          f"{'AutoWS run-var':>15} {'TLX run-var':>12} {'AutoWS vs TLX':>14}")
    for j, n in enumerate(names):
        print(f"{n:6} {aws_err[j]:14.4e} {tlx_err[j]:12.4e} "
              f"{aws_var[j]:15.4e} {tlx_var[j]:12.4e} {cross[j]:14.4e}")
    dq_err = (aws_runs[0][0].float() - rdq.float()).abs()
    adq = aws_runs[0][0].float()
    rdqf = rdq.float()
    print("dQ half errors: "
          f"M0={dq_err[:, :, :N_CTX // 2].max().item():.4e} "
          f"M1={dq_err[:, :, N_CTX // 2:].max().item():.4e} "
          f"N0={dq_err[..., :HEAD_DIM // 2].max().item():.4e} "
          f"N1={dq_err[..., HEAD_DIM // 2:].max().item():.4e}")
    print(f"dQ norms: AutoWS={adq.norm().item():.4e} ref={rdqf.norm().item():.4e} "
          f"vs_2ref={(adq - 2 * rdqf).abs().max().item():.4e}")

    tol = 1e-2
    aws_ok = all(e < tol for e in aws_err)
    tlx_ok = all(e < tol for e in tlx_err)
    print(f"\nAutoWS pass (<{tol}): {aws_ok}   TLX pass (<{tol}): {tlx_ok}")
    if any(v > 1e-3 for v in aws_var):
        print("WARNING: AutoWS backward is NON-DETERMINISTIC across runs (race).")
    if any(v > 1e-3 for v in tlx_var):
        print("WARNING: TLX backward is NON-DETERMINISTIC across runs.")


if __name__ == "__main__":
    main()
