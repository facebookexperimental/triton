---
name: tlx-barrier-hotspots
description: Find likely TLX barrier performance and debugging locations with opt-in read-only compiler remarks. Check phase and async-memory semantics before recommending changes; do not treat remarks as safe named-barrier promotion.
---

# TLX barrier hotspots

Read the shared [workflow](../../../.llms/skills/tlx-barrier-hotspots/references/workflow.md)
for the flag, validation tools, and memory-semantics limits. This diagnostic
does not rewrite IR or establish a performance bottleneck by itself.
