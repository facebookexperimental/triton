// RUN: triton-opt -allocate-proton-shared-memory %s | FileCheck %s
// RUN: triton-opt -allocate-proton-shared-memory -allocate-proton-shared-memory %s | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// A legacy pipeline without the initial report must not gain a partial report.
// CHECK-NOT: ttg.shared_memory_breakdown
// CHECK: module attributes
// CHECK-SAME: ttg.shared = 136 : i32
// CHECK-NOT: ttg.shared_memory_breakdown
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.shared = 1 : i32} {
  tt.func @legacy() {
    // CHECK: ttg.local_alloc {{.*}}allocation.offset = 0 : i32
    %kernel = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi8, #shared, #ttg.shared_memory, mutable>
    // CHECK: ttg.local_alloc {{.*}}allocation.offset = 8 : i32
    %profile = ttg.local_alloc : () -> !ttg.memdesc<16xi64, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %kernel : !ttg.memdesc<1xi8, #shared, #ttg.shared_memory, mutable>
    ttg.local_dealloc %profile : !ttg.memdesc<16xi64, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}
// CHECK-NOT: ttg.shared_memory_breakdown
