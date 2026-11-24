# AMD IR Live-Range Interpretation

Apply this guidance only when `analyze-amd-ir-live-ranges` was explicitly selected and the current measurements or frozen target guidance provide a concrete AMD compiler artifact. Do not attempt to generate a new artifact or report from the isolated candidate workspace. If no artifact is available, use the ordinary AMD optimization guide and choose a hypothesis supported by the supplied measurements.

Pin the kernel specialization and artifact stage before interpreting lifetimes. Preserve textual operation order, audit parser warnings or unresolved SSA values recorded with the artifact, and inspect the original operations around every surprising interval. The accompanying interpretation reference is injected immediately after this guide.

For LDS, distinguish each logical allocation root from its views, slices, and aliases, treating the root interval as the union of all alias uses. For tensors, distinguish definition-to-last-use textual intervals from physical register allocation. Compare pass stages only for the same specialization and filters, and attribute a change only after checking the relevant IR difference.

Use the evidence to form one source-level candidate: delay a load, consume a value earlier, split a scope, reuse LDS after the final alias use, or shorten an accumulator chain. Preserve dependencies, waits, barriers, and source semantics. Correlate TTGIR findings with compiler resource metadata or machine-level evidence before claiming actual VGPR pressure, occupancy, spills, physical LDS overlap, or bank behavior.
