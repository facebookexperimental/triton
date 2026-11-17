// RUN: triton-opt %s --split-input-file --allocate-shared-memory-nv="compute-capability=90 ptx-version=80" --triton-nvidia-gpu-cluster-barrier-mbar-allocator | FileCheck %s
// RUN: triton-opt %s --split-input-file --triton-nvidia-gpu-cluster-barrier-mbar-allocator | FileCheck %s --check-prefix=LEGACY

// Report the allocator's 4-byte WS state array, alignment, and two distinct
// region barriers. Two barriers in the same region share one reservation.
// CHECK: ttg.shared = 72 : i32
// CHECK-SAME: \22kind\22:\22scratch\22
// CHECK-SAME: \22operation\22:\22tt.func\22
// CHECK-SAME: \22size\22:4
// CHECK-SAME: \22name\22:\22late_reservations\22
// CHECK-SAME: \22reserved_bytes\22:4
// CHECK-SAME: \22kind\22:\22alignment\22
// CHECK-SAME: \22offset\22:4
// CHECK-SAME: \22size\22:4
// CHECK-SAME: \22kind\22:\22cluster_barrier\22
// CHECK-SAME: \22offset\22:8
// CHECK-SAME: \22size\22:32
// CHECK-SAME: \22kind\22:\22cluster_barrier\22
// CHECK-SAME: \22offset\22:40
// CHECK-SAME: \22size\22:32
// CHECK-SAME: \22reserved_bytes\22:72
// LEGACY: module attributes
// LEGACY-SAME: ttg.shared = 64 : i32
// LEGACY-NOT: ttg.shared_memory_breakdown
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @late_reservations() {
    ttg.warp_specialize()
    default {
      ttng.cluster_barrier
      ttng.cluster_barrier
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttng.cluster_barrier
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// No cluster barrier means no alignment reservation is made.
// CHECK: ttg.shared = 1 : i32
// CHECK-SAME: \22name\22:\22no_cluster_barrier\22
// CHECK-SAME: \22reserved_bytes\22:1
// CHECK-SAME: \22reservations\22:[]
// CHECK-SAME: \22reserved_bytes\22:1
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @no_cluster_barrier() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<1xi8, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// No cluster barriers preserve the larger allocator footprint.
// CHECK: ttg.shared = 256 : i32
// CHECK-SAME: \22name\22:\22already_large\22
// CHECK-SAME: \22reserved_bytes\22:256
// CHECK-SAME: \22reservations\22:[]
// CHECK-SAME: \22reserved_bytes\22:256
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @already_large() {
    %a = ttg.local_alloc : () -> !ttg.memdesc<256xi8, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// An empty kernel reserves no shared memory.
// CHECK: ttg.shared = 0 : i32
// CHECK-SAME: \22allocations\22:[]
// CHECK-SAME: \22name\22:\22empty_kernel\22
// CHECK-SAME: \22reserved_bytes\22:0
// CHECK-SAME: \22reservations\22:[]
// CHECK-SAME: \22reserved_bytes\22:0
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @empty_kernel() {
    tt.return
  }
}
