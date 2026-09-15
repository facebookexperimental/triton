# NVIDIA Async TMA Output Publication

Use this guidance only for NVIDIA kernels that publish output through asynchronous descriptor stores. Consult `.claude/skills/tlx-api-reference/SKILL.md` for TLX APIs and `.claude/skills/proxy-fence-insertion/SKILL.md` for proxy-fence correctness.

## Look For Serialized Publication

Inspect output paths for independent `tlx.async_descriptor_store` operations that are immediately followed by `tlx.async_descriptor_store_wait(0)`. When output tiles and buffers are independent, consider issuing multiple stores before waiting so publication can overlap.

Use the wait argument as an in-flight-store contract, not as a generic synchronization hint. Before releasing or reusing each local output buffer, prove that the store reading that buffer has completed.

## Preserve Visibility And Lifetime

Prove the complete chain:

1. the producing task finishes writing the local output tile;
2. required async-proxy visibility is established;
3. the descriptor store is issued with valid coordinates and shape;
4. completion is observed before the local buffer is overwritten;
5. producer/consumer barriers are signaled in the correct phase.

Do not remove `tlx.fence("async_shared")` merely because a nearby kernel omits it. Remove a fence only when the producing operation or API contract already provides equivalent visibility, and verify that claim against the generated IR or architecture documentation.

## Batch Stores Conservatively

For multiple output buffers, issue independent stores first and then wait in an order consistent with the permitted outstanding-store count and buffer reuse order. Avoid converting a pipelined output path into an unbounded queue or releasing all buffers after only one store completes.

For write-once output tiles, `eviction_policy="evict_first"` may reduce cache pollution. Treat it as an evidence-driven hint, not a correctness mechanism, and verify performance across protected paths.

## Validate Failure-Prone Cases

Test invalid/dead tiles, boundary coordinates, multiple output slices, persistent-loop reuse, and multi-CTA ownership. If publication changes synchronization or store ordering, require a producer/consumer lifetime proof and reject the candidate on any nondeterministic correctness failure.
