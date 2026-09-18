// RUN: triton-opt \
// RUN:   %S/Inputs/d120-rmsnorm-gemm-post-buffer-allocation.mlir \
// RUN:   -allow-unregistered-dialect \
// RUN:   --nvgpu-test-ws-memory-planner="num-buffers=3 smem-budget=232448" \
// RUN:   -mlir-print-debuginfo -mlir-use-nameloc-as-prefix | \
// RUN:   FileCheck %s --check-prefix=HEURISTIC
// RUN: env TRITON_WS_SMEM_PLAN_TOPK=3 TRITON_WS_SMEM_PLAN_PICK=0 \
// RUN:   TRITON_WS_TMEM_PLAN_TOPK=1 TRITON_WS_TMEM_PLAN_PICK=0 \
// RUN:   triton-opt %S/Inputs/d120-rmsnorm-gemm-post-buffer-allocation.mlir \
// RUN:   -allow-unregistered-dialect \
// RUN:   --nvgpu-test-ws-memory-planner="num-buffers=3 smem-budget=232448 smem-plan-search" \
// RUN:   -mlir-print-debuginfo -mlir-use-nameloc-as-prefix | \
// RUN:   FileCheck %s --check-prefix=SEARCH0
// RUN: env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=1 \
// RUN:   triton-opt %S/Inputs/d120-rmsnorm-gemm-post-buffer-allocation.mlir \
// RUN:   -allow-unregistered-dialect \
// RUN:   --nvgpu-test-ws-memory-planner="num-buffers=3 smem-budget=232448 smem-plan-search" \
// RUN:   -mlir-print-debuginfo -mlir-use-nameloc-as-prefix | \
// RUN:   FileCheck %s --check-prefix=SEARCH1
// RUN: env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=2 \
// RUN:   triton-opt %S/Inputs/d120-rmsnorm-gemm-post-buffer-allocation.mlir \
// RUN:   -allow-unregistered-dialect \
// RUN:   --nvgpu-test-ws-memory-planner="num-buffers=3 smem-budget=232448 smem-plan-search" \
// RUN:   -mlir-print-debuginfo -mlir-use-nameloc-as-prefix | \
// RUN:   FileCheck %s --check-prefix=SEARCH2
// RUN: rm -f %t
// RUN: env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=0 \
// RUN:   TRITON_WS_MEM_PLAN_TOPK_DUMP=%t \
// RUN:   triton-opt %S/Inputs/d120-rmsnorm-gemm-post-buffer-allocation.mlir \
// RUN:   -allow-unregistered-dialect \
// RUN:   --nvgpu-test-ws-memory-planner="num-buffers=3 smem-budget=232448 smem-plan-search" \
// RUN:   -o /dev/null
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t
// RUN: env TRITON_WS_SMEM_PLAN_TOPK=3 TRITON_WS_SMEM_PLAN_PICK=0 \
// RUN:   TRITON_WS_TMEM_PLAN_TOPK=1 TRITON_WS_TMEM_PLAN_PICK=0 \
// RUN:   triton-opt %S/Inputs/d120-rmsnorm-gemm-post-buffer-allocation.mlir \
// RUN:   -allow-unregistered-dialect \
// RUN:   --nvgpu-test-ws-memory-planner="num-buffers=3 smem-budget=232448 smem-plan-search" \
// RUN:   --nvgpu-warp-specialization="num-stages=3 smem-budget=232448" | \
// RUN:   FileCheck %s --check-prefix=CODEPART
// RUN: env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=2 \
// RUN:   triton-opt %S/Inputs/d120-rmsnorm-gemm-post-buffer-allocation.mlir \
// RUN:   -allow-unregistered-dialect \
// RUN:   --nvgpu-test-ws-memory-planner="num-buffers=3 smem-budget=232448 smem-plan-search" \
// RUN:   --nvgpu-test-ws-code-partition="num-buffers=3 channel-cycle-audit=true" | \
// RUN:   FileCheck %s --check-prefix=CYCLE-POSITIVE
// RUN: sed \
// RUN:   -e '/nvws.descriptor_load %a_desc/s/loop.cluster = 1 : i32, loop.stage = 0 : i32/loop.cluster = 0 : i32, loop.stage = 1 : i32/' \
// RUN:   -e '/%a_47 = ttg.local_load/s/loop.cluster = 2 : i32, loop.stage = 0 : i32/loop.cluster = 1 : i32, loop.stage = 1 : i32/' \
// RUN:   -e '/ttg.local_store %a_47/s/loop.cluster = 2 : i32, loop.stage = 0 : i32/loop.cluster = 1 : i32, loop.stage = 1 : i32/' \
// RUN:   -e '/nvws.descriptor_load %b_desc/s/loop.cluster = 0 : i32, loop.stage = 1 : i32/loop.cluster = 1 : i32, loop.stage = 0 : i32/' \
// RUN:   -e '/%a_f32 = arith.extf/s/loop.cluster = 2 : i32, loop.stage = 0 : i32/loop.cluster = 1 : i32, loop.stage = 1 : i32/' \
// RUN:   -e '/loc(#loc65)/s/loop.cluster = 3 : i32, loop.stage = 0 : i32/loop.cluster = 2 : i32, loop.stage = 1 : i32/' \
// RUN:   -e '/}) {async_task_id = array<i32: 0>, loop.cluster = 4/s/loop.cluster = 4 : i32, loop.stage = 0 : i32/loop.cluster = 3 : i32, loop.stage = 1 : i32/' \
// RUN:   -e '/loc(#loc67)/s/loop.cluster = 1 : i32, loop.stage = 1 : i32/loop.cluster = 4 : i32, loop.stage = 1 : i32/' \
// RUN:   -e '/%acc_51 = ttg.memdesc_trans/s/loop.cluster = 1 : i32, loop.stage = 1 : i32/loop.cluster = 2 : i32, loop.stage = 0 : i32/' \
// RUN:   %S/Inputs/d120-rmsnorm-gemm-post-buffer-allocation.mlir > %t.bearly.mlir
// RUN: not env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=1 \
// RUN:   triton-opt %t.bearly.mlir \
// RUN:   -allow-unregistered-dialect \
// RUN:   --nvgpu-test-ws-memory-planner="num-buffers=3 smem-budget=232448 smem-plan-search" \
// RUN:   --nvgpu-test-ws-code-partition="num-buffers=3 channel-cycle-audit=true" 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CYCLE-REJECT
// RUN: not env TRITON_WS_MEM_PLAN_TOPK=3 TRITON_WS_MEM_PLAN_PICK=2 \
// RUN:   triton-opt %t.bearly.mlir \
// RUN:   -allow-unregistered-dialect \
// RUN:   --nvgpu-test-ws-memory-planner="num-buffers=3 smem-budget=232448 smem-plan-search" \
// RUN:   --nvgpu-test-ws-code-partition="num-buffers=3 channel-cycle-audit=true" 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CYCLE-REJECT

// Production-shaped D120426461 memory-planner oracle. A feeds both the RMS
// reduction and the MMA, while B feeds only the MMA. The existing heuristic
// therefore assigns A3/B2. Eight output subtiles share one three-copy staging
// ring. Fixed-group search preserves that ring and exposes A2/B2 and A2/B3 as
// the first two alternatives without operand-specific annotations.
// The checked-in loop schedule is the safe A-early rank: A/relay is at stage
// 0, B is at stage 1, and the MMA is at stage 2. The sed pipeline above changes
// only those captured schedule coordinates to reconstruct the B-early rank.

// A-early/A2-B3 passes the gate. Specialized output-staging and TMEM channels
// remain explicitly unsupported, so the overall coverage status is
// "unsupported" rather than "safe"; crucially, no supported SCC is unsafe.
// CYCLE-POSITIVE-LABEL: tt.func public @d120_rmsnorm_gemm
// CYCLE-POSITIVE-SAME: nvws.test.channel_cycle_edge_count = 28 : i64
// CYCLE-POSITIVE-SAME: nvws.test.channel_cycle_event_count = 12 : i64
// CYCLE-POSITIVE-SAME: nvws.test.channel_cycle_status = "unsupported"
// CYCLE-POSITIVE-SAME: nvws.test.channel_cycle_supported_channels = 3 : i64
// CYCLE-POSITIVE-SAME: nvws.test.channel_cycle_unsupported_channels = 10 : i64

// The B-early reconstruction is rejected for both A2/B2 and A2/B3 with the
// same zero-credit A-relay/B witness.
// Specialized output-staging and TMEM channels remain explicitly unsupported
// in this first slice, but an unsafe supported SCC takes priority.
// CYCLE-REJECT: error: warp specialization rejected an unsafe post-memory channel protocol: total iteration distance 0, channels [1, 1, 4, 4, 4, 1], edge distances [0, 0, 1, 0, -1, 0]

// HEURISTIC-LABEL: tt.func public @d120_rmsnorm_gemm
// HEURISTIC: %a = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32}
// HEURISTIC: %acc = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32}
// HEURISTIC: %a_5 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32}
// HEURISTIC-COUNT-8: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 3 : i32, buffer.tmaStaging = 1 : i32}

// SEARCH0-LABEL: tt.func public @d120_rmsnorm_gemm
// SEARCH0: %a = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32}
// SEARCH0: %acc = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32}
// SEARCH0: %a_5 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32}
// SEARCH0-COUNT-8: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 3 : i32, buffer.tmaStaging = 1 : i32}

// SEARCH1-LABEL: tt.func public @d120_rmsnorm_gemm
// SEARCH1: %a = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32}
// SEARCH1: %acc = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32}
// SEARCH1: %a_5 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32}
// SEARCH1-COUNT-8: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 3 : i32, buffer.tmaStaging = 1 : i32}

// SEARCH2-LABEL: tt.func public @d120_rmsnorm_gemm
// SEARCH2: %a = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32}
// SEARCH2: %acc = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 1 : i32}
// SEARCH2: %a_5 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32}
// SEARCH2-COUNT-8: ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 3 : i32, buffer.tmaStaging = 1 : i32}

// MANIFEST: {"kind": "memory", "schedule_pick": 0, "pool": "smem-fixed", "rank": 0, "selected": true, "score": {{[0-9.]+}}, "blocks": [{"id": 0, "copy": 3, "members": 1, "allocated": true}, {"id": 1, "copy": 2, "members": 1, "allocated": true}, {"id": 2, "copy": 1, "members": 1, "allocated": true}, {"id": 3, "copy": 3, "members": 8, "allocated": true}]}
// MANIFEST-NEXT: {"kind": "memory", "schedule_pick": 0, "pool": "smem-fixed", "rank": 1, "selected": false, "score": {{[0-9.]+}}, "blocks": [{"id": 0, "copy": 2, "members": 1, "allocated": true}, {"id": 1, "copy": 2, "members": 1, "allocated": true}, {"id": 2, "copy": 1, "members": 1, "allocated": true}, {"id": 3, "copy": 3, "members": 8, "allocated": true}]}
// MANIFEST-NEXT: {"kind": "memory", "schedule_pick": 0, "pool": "smem-fixed", "rank": 2, "selected": false, "score": {{[0-9.]+}}, "blocks": [{"id": 0, "copy": 2, "members": 1, "allocated": true}, {"id": 1, "copy": 3, "members": 1, "allocated": true}, {"id": 2, "copy": 1, "members": 1, "allocated": true}, {"id": 3, "copy": 3, "members": 8, "allocated": true}]}

// A direct-grid epilogue has no outer loop to carry an accumulation counter.
// Its eight straight-line staging channels must nevertheless rotate through
// the three physical slots in program order. Each slot occurs twice below:
// once for the producer local_store and once for the consumer TMA store.
// CODEPART-LABEL: tt.func public @d120_rmsnorm_gemm
// CODEPART: %[[STAGING:.*]] = ttg.local_alloc {{.*}}buffer.copy = 3 : i32, buffer.id = 3 : i32{{.*}} : () -> !ttg.memdesc<3x128x16xbf16
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c0_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c0_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c1_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c1_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c2_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c2_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c0_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c0_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c1_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c1_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c2_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c2_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c0_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c0_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c1_i32]
// CODEPART-NEXT: ttg.memdesc_index %[[STAGING]][%c1_i32]
