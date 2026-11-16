# Hopper Machine Limits

Use this guidance only for NVIDIA Hopper (H100, sm_90) targets. Consult these limits before proposing tile sizes, stage counts, warp budgets, or persistent-grid geometry, and prove the candidate fits them.

## Recognize When The Budget Binds

Route to this document when any of these hold:

- Codegen fails with `OutOfResources: shared memory required X > 232448`.
- NCU names `Block_Limit_Shared_Mem` or `Block_Limit_Regs` as the binding limiter while occupancy sits well below the 32-block and 64-warp ceilings.
- NCU shows HMMA instead of WGMMA on a sm_90 dot, or tensor-pipe utilization collapses after a tile shrink.
- `total_tiles / NUM_SMS` is near 1.0 with the long-scoreboard stall dominant: per-CTA footprint, not pipeline depth, is the lever.

Do not deepen the pipeline or widen the tile until the binding budget is relieved.

## Check The Hard Limits First

These limits fail codegen when exceeded. They are preconditions, not tuning knobs.

| Limit | Value |
|---|---|
| Dynamic SMEM cap (per CTA, opt-in) | 232,448 bytes (exactly 227 KiB) |
| SMEM residency budget (per SM) | ~228 KiB shared by all resident CTAs |
| WGMMA minimum tile | m=64; k=16 (fp16/bf16), k=32 (fp8) |

Failure modes:

- SMEM over budget fails codegen with `OutOfResources: shared memory required X > 232448`; not recoverable by retry. Keep a 2–4 KiB safety margin below the cap when budgeting GEMM-style tiles.
- Fitting the per-CTA cap does not imply co-residency: two 128 KiB CTAs each fit the 232,448-byte per-CTA limit, but their 256 KiB sum exceeds the ~228 KiB per-SM SMEM budget, so the runtime occupancy result may allow only one resident block. Size resident-block counts from `Block_Limit_Shared_Mem` or measured occupancy, not from naive division of the per-CTA cap.
- A dot below the WGMMA minimums does not use the WGMMA pipe: it falls back to HMMA or fails to compile, and tensor-pipe utilization collapses. Never shrink M below 64 or K below the dtype minimum to relieve SMEM or registers.

## Budget The Soft Limits

These limits reduce occupancy when exceeded, but the kernel still runs.

| Limit | Value |
|---|---|
| Compute capability | 9.0 (sm_90) |
| Unified L1 / Texture / SMEM pool (per SM) | 256 KiB (unified carve-out, not the SMEM residency ceiling) |
| SMEM bank structure | 32 banks × 4 bytes |
| L2 cache | 50 MB (all H100 SKUs) |
| Register file (per SM) | 65,536 32-bit registers |
| Max threads (per SM) | 2048 (64 warps) |
| Max blocks (per SM) | 32 |

Failure modes:

- Register over budget spills to local memory or binds `Block_Limit_Regs`, cutting resident blocks; relieve it before deepening the pipeline.
- SMEM over budget binds `Block_Limit_Shared_Mem` at one resident block; halve the largest buffer axis before adding stages.
- Working sets past 50 MB L2 spill to DRAM and the kernel goes bandwidth-bound; restructure reuse before tuning the tile.
- Growing the SMEM reservation shrinks the L1 carve-out from the shared 256 KiB pool; L1-sensitive kernels regress.
- Unswizzled accesses to the 32-bank SMEM collide and serialize; swizzle or transpose the layout instead of shrinking the tile.

Show the occupancy math in the candidate rationale: per-CTA registers, per-CTA SMEM, and the resulting resident-block count against these ceilings.

## Handle SKU Variance

SM count and DRAM bandwidth vary across H100 SKUs (SXM5 HBM3 vs PCIe HBM2e, MIG partitions). Never hardcode them.

- Query the SM count at runtime (`torch.cuda.get_device_properties(...).multi_processor_count`) for wave-count math (`total_tiles / NUM_SMS`); a hardcoded count oversubscribes smaller parts and idles SMs on larger ones.
- Measure effective bandwidth on the target with the profiler instead of assuming a nameplate number; roofline claims require measured bytes and measured time from the same run, or a bandwidth-bound kernel looks under-roofline and sends the optimizer after phantom compute wins.

## Validate The Budget

For every candidate that changes tile geometry, stage depth, or warp budget, retain evidence that the budget holds: per-CTA SMEM and register totals with the safety margin applied, NCU limiter and occupancy readings before and after with the trigger signature cleared, and stable end-to-end benchmark latency. Treat fitting the limits as necessary but not sufficient; promotion still requires a measured win with passing correctness.
