// RUN: env STANDALONE_MODULO=1 TRITON_USE_MODULO_SCHEDULE=contracted \
// RUN:   TRITON_MODULO_TOPK=4 TRITON_MODULO_PICK=0 \
// RUN:   triton-opt %S/Inputs/d120-rmsnorm-gemm-pre-modulo.mlir \
// RUN:   -allow-unregistered-dialect -nvgpu-modulo-schedule \
// RUN:   -mlir-print-debuginfo -mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=BASELINE
// RUN: env STANDALONE_MODULO=1 TRITON_USE_MODULO_SCHEDULE=contracted \
// RUN:   TRITON_MODULO_TOPK=4 TRITON_MODULO_PICK=1 \
// RUN:   triton-opt %S/Inputs/d120-rmsnorm-gemm-pre-modulo.mlir \
// RUN:   -allow-unregistered-dialect -nvgpu-modulo-schedule \
// RUN:   -mlir-print-debuginfo -mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=B-EARLY
// RUN: rm -f %t
// RUN: env STANDALONE_MODULO=1 TRITON_USE_MODULO_SCHEDULE=contracted \
// RUN:   TRITON_MODULO_TOPK=4 TRITON_MODULO_PICK=0 \
// RUN:   TRITON_WS_SEARCH_MANIFEST=%t \
// RUN:   triton-opt %S/Inputs/d120-rmsnorm-gemm-pre-modulo.mlir \
// RUN:   -allow-unregistered-dialect -nvgpu-modulo-schedule > /dev/null
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t

// Production-shaped D120426461 oracle captured from a bf16 fused RMSNorm +
// GEMM with 128x128x128 tiles and an eight-way output subtile. Unlike the
// reduced one-MMA oracle, A also feeds the row sum-of-squares reduction.
// There are no operand-depth or per-operation schedule annotations in the
// input; the Contracted scheduler owns loop.stage and loop.cluster.

// Rank zero keeps source load order. Both operands are one logical iteration
// ahead of the MMA; A's reduction path remains in the original DDG.
// BASELINE-LABEL: tt.func public @d120_rmsnorm_gemm
// BASELINE: %[[A:.*]] = tt.descriptor_load {{.*}}loop.cluster = 1 : i32, loop.stage = 0 : i32{{.*}}loc("a"
// BASELINE: %[[B:.*]] = tt.descriptor_load {{.*}}loop.cluster = 2 : i32, loop.stage = 0 : i32{{.*}}loc("b"
// BASELINE: "tt.reduce"
// BASELINE: ttng.tc_gen5_mma {{.*}}loop.cluster = 0 : i32, loop.stage = 1 : i32

// Rank one prioritizes B's address/load slice without bypassing the A -> RMS
// reduction dependency. Both loads still occupy stage zero on this real graph.
// B-EARLY-LABEL: tt.func public @d120_rmsnorm_gemm
// B-EARLY: %[[A:.*]] = tt.descriptor_load {{.*}}loop.cluster = 2 : i32, loop.stage = 0 : i32{{.*}}loc("a"
// B-EARLY: %[[B:.*]] = tt.descriptor_load {{.*}}loop.cluster = 1 : i32, loop.stage = 0 : i32{{.*}}loc("b"
// B-EARLY: "tt.reduce"
// B-EARLY: ttng.tc_gen5_mma {{.*}}loop.cluster = 0 : i32, loop.stage = 1 : i32

// The frontier contains both load orders and larger-II representatives.
// MANIFEST: {"kind": "schedule", "rank": 0, "selected": true, "ii": 5, "load_order_variant": 0, "signature": [1, 0, 0, 1, 0, 2]}
// MANIFEST-NEXT: {"kind": "schedule", "rank": 1, "selected": false, "ii": 5, "load_order_variant": 1, "signature": [1, 0, 0, 2, 0, 1]}
// MANIFEST-NEXT: {"kind": "schedule", "rank": 2, "selected": false, "ii": 6, "load_order_variant": 0, "signature": [1, 0, 0, 1, 0, 2]}
// MANIFEST-NEXT: {"kind": "schedule", "rank": 3, "selected": false, "ii": 15, "load_order_variant": 0, "signature": [1, 0, 0, 1, 0, 2]}
