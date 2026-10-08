# NVIDIA Memory-Bound Kernels And Autotune Selection

Apply this guidance to NVIDIA kernels whose profile points at DRAM, typically GEMMs with a small reduction dimension or a narrow output dimension, and to any kernel whose host code selects configurations through Triton autotuning.

## Classify Before Editing

Measure DRAM bytes read and written (`dram__bytes_read.sum`, `dram__bytes_write.sum`) and compare them with the algorithmic minimum: each input element read once, each output element written once. Also compare them with a reference implementation on the same shape. tritonbench's `--metrics ncu_rep` profiles only the timed call, so autotuning launches stay out of the report.

- If the kernel moves more than the minimum, look for missing on-chip reuse first. Pipeline or epilogue changes cannot remove redundant DRAM traffic.
- If the kernel already moves the minimum and DRAM throughput is high, it is bandwidth-bound. The remaining gap to a reference is DRAM efficiency. Compute-side changes (deeper rings, async epilogues, wait relaxation) are unlikely to help; candidates should change which data stays in L2.
- A quick compute-bound check: profilers lock SM clocks, often far below boost. If the profiled duration barely changes at the lower clock, the kernel is not compute-bound.

## Shape L2 Reuse

When several consecutive work items read the same input panel, schedule them back to back. For tile-grouped GEMM scheduling, a group size of 1 (row-major tile order) makes all output tiles of a narrow output dimension consume one input panel together. Wider groups still suit square problems, so add the narrow order to the search space instead of replacing the existing one.

When an input panel is re-read within a short window, `eviction_policy="evict_last"` on its loads can keep it in L2 for the repeat reads. Gate the hint by the reuse pattern, for example on a small output dimension where only a few tiles share each panel. Applied unconditionally, it keeps panels resident long after their reuse ends and can regress large problems.

Cache hints are not free in either direction. `evict_first` on write-once outputs, or on streamed inputs, can lower throughput when the hardware's default policy was already serving later reads. Treat every hint as a hypothesis, measure it per shape class, and keep it only where it wins.

## Keep Autotune Selections Stable

Autotuning measures each configuration briefly. Some configurations are bimodal on particular shapes: pinned and measured repeatedly, they run consistently slow, yet they occasionally time as fastest during tuning. Autotune can then lock in a configuration much slower than the best one.

When a full-suite result is worse than the same shape measured with its best configuration pinned:

1. Print the selected configuration and the sorted per-configuration timings (`TRITON_PRINT_AUTOTUNING=1`) across several runs.
2. Pin each candidate configuration and measure it repeatedly to find its true throughput.
3. Prune configurations that are slow or unstable for an identifiable shape class with `early_config_prune`. Return the original list if the filter would leave nothing, so explicitly pinned configurations still work.

A larger search space improves the best reachable configuration but gives a noisy candidate more chances to win. Prefer adding configurations that cover a distinct regime, and prune them where they are known to lose.

## Validate

Confirm that rebuilt artifacts actually contain the edited source before comparing results; an unchanged binary silently reproduces the old numbers. Compare full-suite runs, not only single-shape runs, because selection noise and L2 state differ between them. Report pinned and autotuned results separately when they disagree.
