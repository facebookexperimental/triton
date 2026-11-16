# C Launcher Eligibility

Host-side launch overhead can dominate small-shape runtime on both NVIDIA and AMD. A TLX WS `addmm` kernel on H100 spent ~58us of host work per call against a ~30us kernel, so `do_bench` reported 30us while sustained wall time was 58us. The single largest avoidable contributor (~18us/call) was a module-level global that disabled the C fast-dispatch launcher.

The C dispatcher applies to both backends: each target builds its own via the `triton_dispatcher_factory` of its backend package (`triton.backends.nvidia` or `triton.backends.amd`), gated on the target reporting `cuda` or `hip`. The eligibility rules below are identical for both.

## Keep @triton.jit Kernels Free Of Module Globals

The C launcher (`native_fast_dispatch_insert`) applies only when the kernel references no Python globals, i.e. `kernel.used_global_vals` is empty. Any module-level constant read inside the kernel — even an innocent `tl.constexpr`-typed configuration value — silently forces the slow Python dispatch path on every launch.

Rules:

- Never read a module-level Python variable from inside a `@triton.jit` kernel. Pass configuration (stages, flags, sizes) as explicit `tl.constexpr` kernel arguments, or define the constant inside the kernel body.
- Treat this as a launch-path constraint, not a style preference: one stray global costs ~10-20us per call on H100, which exceeds the kernel time of small GEMM shapes. The penalty applies equally to HIP targets since they share the same dispatch gating.
- Audit helper call chains: a global read in any function inlined into the kernel counts. Check transitive callees, not just the kernel body.

## Verify Eligibility

- After JIT, assert `kernel.used_global_vals` is empty for performance-critical kernels.
- Watch for the "C launcher cannot be used" warning at first launch; its presence means the kernel fell back to slow dispatch.
- Confirm the fast path is active via `_get_jit_cache_proxy` / `_fc_seeded` state when diagnosing regressions.

## Measure Host Overhead, Not Just Kernel Time

- `do_bench` reports GPU spans and hides host work behind a deep queue: a kernel can report 30us while each call costs 58us of wall time.
- For small shapes, measure sustained wall time (N back-to-back calls, wall / N) and compare against `do_bench`. A large gap means the kernel is host-bound and launch-path changes outweigh further kernel tuning.
- CUDA/ROCm-graph replay removes host work and isolates kernel time; use eager-vs-graph deltas to split the two before tuning the kernel body.

## Remaining Overhead Sources

Once C-launcher eligible, the next contributors are per-call descriptor rebuilds (~7-9us for 3 TMA descriptors on CUDA) and dispatch argument processing plus per-launch descriptor fills (~10us). Prefer caching descriptors across calls and hoisting fills out of the per-launch path over further kernel-body tuning while wall time still exceeds kernel time.
