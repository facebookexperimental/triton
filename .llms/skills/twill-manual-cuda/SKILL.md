---
name: twill-manual-cuda
description: >
  Guide agent-assisted manual B200 CUDA C++ lowering from validated
  twill-pipelined-warp-ir-v1 artifacts, with explicit clean-room provenance,
  mapping, memory, synchronization, correctness, and performance contracts.
metadata:
  oncalls: triton
  strict: true
  apply_to_path: 'third-party/triton/beta/triton/third_party/tlx/tools/paper_joint_solver/.*'
  apply_to_user_prompt: '(?i)(twill.*(manual|cuda|b200|lower)|manual.*cuda.*twill|twill-pipelined-warp-ir-v1|agent-assisted-manual-cuda)'
  apply_to_content: 'twill-(pipelined-warp-ir|manual-cuda-handoff)-v1'
---

# Twill Manual B200 CUDA

Use this skill only for an expert-reviewed, agent-assisted manual CUDA C++
implementation starting from a validated `twill-pipelined-warp-ir-v1`
artifact. It is not an automatic code generator and does not turn a metadata
round trip into executable-lowering evidence.

Read the references progressively:

- Before drafting code or opening a comparison implementation, read
  [references/workflow.md](references/workflow.md). It defines the clean-room
  boundary, required artifacts, review gates, and allowed claims.
- When assigning every IR instruction to CUDA source and physical execution,
  read [references/mapping-contract.md](references/mapping-contract.md).
- Before selecting TMA, TCGEN05, TMEM, fence, or barrier operations, read
  [references/b200-primitives.md](references/b200-primitives.md).
- Before allocating registers, shared memory, TMEM, descriptors, or ring
  slots, read [references/memory-plan.md](references/memory-plan.md).
- Before implementing or changing any synchronization, read
  [references/sync-phase-audit.md](references/sync-phase-audit.md).
- After the first build and before interpreting performance, read
  [references/correctness-performance.md](references/correctness-performance.md).
- Before requesting review or making provenance/performance claims, read
  [references/presubmit.md](references/presubmit.md).

The implementation must carry `authoring_mode=agent-assisted-manual-cuda` and
checked-in mapping, memory, and synchronization manifests. FA4 and FA3 are
black-box correctness/performance baselines only: do not reuse or infer their
warp roles, MMA order, or barrier protocol. The only reusable implementation
exception is a separately reviewed, schedule-neutral leaf wrapper around one
TMA, TCGEN05, or mbarrier primitive, recorded in the provenance manifest.
