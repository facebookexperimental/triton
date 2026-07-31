"""Draw-order counter-example kernel, no warp specialization — dump source
for the case11_wait_order fixture.

Minimal kernel exhibiting the schedule-DRAW blind spot (distilled from the
case4 FA-bwd 2026-07-21 draw calibration): per iteration, two independent
MMAs whose TMEM results are both read by software; a long SFU chain depends
only on the first read; the second read feeds only a cheap combine at the
end. The combine (truncated to fp16) is the A-operand of a third,
accumulating MMA — the loop-carried recurrence.

    s1 = x_i @ w1^T          (MMA_A -> TMEM, read early: feeds exp2 chain)
    s2 = y_i @ w2^T          (MMA_B -> TMEM, read DEFERRABLE: feeds combine)
    p  = exp2(s1 * scale)    (long SFU chain)
    z  = (p - s2) -> fp16    (combine + trunc)
    acc += z @ v             (MMA_C, loop-carried)

Dependences admit two order families for read(s2) inside the compute warp's
in-order stream: coalesced with read(s1) before the chain, or deferred until
just before the combine. Both are model-equivalent (same II / objective /
partition); the emitted kernels differ on hardware via TMEM slot-release
timing and TC-pipe serialization (case4 measured 292.8 vs 269.6 TF for the
same axis).

Dump the pre-modulo TTGIR with (single GPU compile, small shape):

    TRITON_ALWAYS_COMPILE=1 TRITON_USE_MODULO_SCHEDULE=1 \
    MLIR_ENABLE_DUMP=wait_order_kernel_nows MLIR_DUMP_PATH=$PWD/dump.mlir \
    env -u LD_LIBRARY_PATH <venv-python> wait_order_nows.py --dump

then extract the module printed immediately before the modulo-schedule pass
("IR Dump Before {anonymous}::ModuloSchedulePass").
"""

import sys

import torch
import triton
import triton.language as tl


@triton.jit
def wait_order_kernel_nows(X, Y, W1, W2, V, Out, scale, T,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           KD: tl.constexpr, ON: tl.constexpr):
    pid = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(X, [T * BLOCK_M * tl.num_programs(0), KD],
                                       [KD, 1], [BLOCK_M, KD])
    y_desc = tl.make_tensor_descriptor(Y, [T * BLOCK_M * tl.num_programs(0), KD],
                                       [KD, 1], [BLOCK_M, KD])
    w1_desc = tl.make_tensor_descriptor(W1, [BLOCK_N, KD], [KD, 1],
                                        [BLOCK_N, KD])
    w2_desc = tl.make_tensor_descriptor(W2, [BLOCK_N, KD], [KD, 1],
                                        [BLOCK_N, KD])
    v_desc = tl.make_tensor_descriptor(V, [BLOCK_N, ON], [ON, 1],
                                       [BLOCK_N, ON])
    out_desc = tl.make_tensor_descriptor(Out, [BLOCK_M * tl.num_programs(0), ON],
                                         [ON, 1], [BLOCK_M, ON])
    w1 = w1_desc.load([0, 0])
    w2 = w2_desc.load([0, 0])
    v = v_desc.load([0, 0])
    w1t = tl.trans(w1)
    w2t = tl.trans(w2)
    acc = tl.zeros([BLOCK_M, ON], tl.float32)
    base = pid * T * BLOCK_M
    for i in range(T):
        off = base + i * BLOCK_M
        x = x_desc.load([off, 0])
        y = y_desc.load([off, 0])
        s1 = tl.dot(x, w1t)
        s2 = tl.dot(y, w2t)
        p = tl.math.exp2(s1 * scale)
        z = p - s2
        acc = tl.dot(z.to(tl.float16), v, acc)
    out_desc.store([pid * BLOCK_M, 0], acc.to(tl.float16))


def run(G=4, T=8, BLOCK_M=128, BLOCK_N=64, KD=64, ON=128, check=True):
    torch.manual_seed(0)
    triton.set_allocator(
        lambda size, align, stream: torch.empty(size, device="cuda",
                                                dtype=torch.int8))
    x = torch.randn(G * T * BLOCK_M, KD, device="cuda", dtype=torch.float16) * 0.25
    y = torch.randn(G * T * BLOCK_M, KD, device="cuda", dtype=torch.float16) * 0.25
    w1 = torch.randn(BLOCK_N, KD, device="cuda", dtype=torch.float16) * 0.25
    w2 = torch.randn(BLOCK_N, KD, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(BLOCK_N, ON, device="cuda", dtype=torch.float16) * 0.25
    out = torch.empty(G * BLOCK_M, ON, device="cuda", dtype=torch.float16)
    scale = 1.0
    wait_order_kernel_nows[(G,)](x, y, w1, w2, v, out, scale, T,
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, KD=KD,
                                 ON=ON, num_warps=4, num_ctas=1, num_stages=2,
                                 maxRegAutoWS=152)
    if not check:
        torch.cuda.synchronize()
        return None
    xf = x.float().view(G, T, BLOCK_M, KD)
    yf = y.float().view(G, T, BLOCK_M, KD)
    ref = torch.zeros(G, BLOCK_M, ON, device="cuda")
    for i in range(T):
        s1 = xf[:, i] @ w1.float().T
        s2 = yf[:, i] @ w2.float().T
        z = (torch.exp2(s1 * scale) - s2).half().float()
        ref += z @ v.float()
    rel = (out.float().view(G, BLOCK_M, ON) - ref).abs().max().item() / \
        ref.abs().max().item()
    nan = int(torch.isnan(out).sum().item())
    print(f"case11 wait_order: rel={rel:.3e} nan={nan} "
          f"{'PASS' if rel < 2e-2 and nan == 0 else 'FAIL'}")
    return rel < 2e-2 and nan == 0


if __name__ == "__main__":
    if "--dump" in sys.argv:
        run(G=2, T=4, check=False)
        print("dump run complete")
    else:
        ok = run()
        sys.exit(0 if ok else 1)
