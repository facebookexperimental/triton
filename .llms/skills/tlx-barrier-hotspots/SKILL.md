---
name: tlx-barrier-hotspots
description: Identify looped TLX barrier waits worth profiling or debugging using read-only compiler remarks; interpret mbarrier phase and async-memory semantics before proposing changes. Use for TLX barrier stalls, suspected synchronization bottlenecks, or barrier-related hangs, not as a named-barrier conversion recipe.
metadata:
  oncalls: triton_dev
  strict: true
  apply_to_user_prompt: '(?i)(tlx.*barrier|barrier.*tlx)'
---

# TLX barrier hotspot triage

When locating or interpreting TLX barrier performance/debugging leads, read
[references/workflow.md](references/workflow.md). The optional diagnostic
reports locations; it does not prove a bottleneck or safe barrier replacement.
