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

// Production-shaped D120426461 memory-planner oracle. A feeds both the RMS
// reduction and the MMA, while B feeds only the MMA. The existing heuristic
// therefore assigns A3/B2. Eight output subtiles share one three-copy staging
// ring. Fixed-group search preserves that ring and exposes A2/B2 and A2/B3 as
// the first two alternatives without operand-specific annotations.

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
