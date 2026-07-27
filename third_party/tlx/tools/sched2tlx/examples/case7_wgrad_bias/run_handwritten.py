"""Case 7 — run the hand-written WS wgrad GEMM + bias-reduce on B200 vs torch."""

from __future__ import annotations

import sys

import handwritten
import torch
import triton


def alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def main() -> int:
    triton.set_allocator(alloc_fn)
    torch.manual_seed(0)
    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

    BLOCK_KO, BLOCK_NI, BLOCK_M = 128, 128, 64

    shapes = [
        (1024, 256, 256),
        (4096, 1024, 1024),
        (8192, 2048, 1024),
        (16384, 1024, 1024),
    ]
    failed = 0
    for M, K_out, N_in in shapes:
        dout = torch.randn(M, K_out, device="cuda", dtype=torch.float16)
        act = torch.randn(M, N_in, device="cuda", dtype=torch.float16)
        dw = torch.full((K_out, N_in), float("nan"), device="cuda", dtype=torch.float16)
        db = torch.full((K_out,), float("nan"), device="cuda", dtype=torch.float32)

        grid = (NUM_SMS,)
        try:
            handwritten.wgrad_bias_ws[grid](
                dout,
                act,
                dw,
                db,
                M,
                K_out,
                N_in,
                BLOCK_KO=BLOCK_KO,
                BLOCK_NI=BLOCK_NI,
                BLOCK_M=BLOCK_M,
                # NUM_SMEM_BUFFERS / NUM_TMEM_BUFFERS + num_warps/num_ctas/num_stages injected by @triton.autotune.
            )
        except Exception as e:
            print(
                f"[FAIL-COMPILE] M={M} K_out={K_out} N_in={N_in}: "
                f"{str(e).splitlines()[-1][:90]}"
            )
            failed += 1
            continue

        dw_ref = (dout.float().T @ act.float()).to(torch.float16)
        db_ref = dout.float().sum(0)
        dw_rel = (dw.float() - dw_ref.float()).abs().max().item() / max(
            dw_ref.float().abs().max().item(), 1e-9
        )
        db_rel = (db - db_ref).abs().max().item() / max(db_ref.abs().max().item(), 1e-9)
        nan = int(torch.isnan(dw).sum().item() + torch.isnan(db).sum().item())
        ok = nan == 0 and dw_rel < 5e-3 and db_rel < 5e-3
        print(
            f"[{'PASS' if ok else 'FAIL'}] M={M} K_out={K_out} N_in={N_in}  nan={nan}  "
            f"dW rel={dw_rel:.3e}  dB rel={db_rel:.3e}"
        )
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
