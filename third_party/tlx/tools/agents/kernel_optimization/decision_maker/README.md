# Decision Maker

The Decision Maker is a deterministic controller. It owns the global loop, budgets, run
state, authoritative correctness and performance evaluation, promotion, stopping, final
revalidation, artifacts, and VCS finalization.

It executes the Optimizer's proposals but never invents optimization hypotheses.

Target harnesses, profiling, and platform metadata live here alongside the controller that
applies their results.

`targets/` discovers manifest-backed harness bundles by operation and architecture.
`profiling/` owns normalized profiling requests, parsers, documentation, and the mapping
from a backend to its native profiler.
