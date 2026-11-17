// RUN: triton-opt --split-input-file -allocate-shared-memory -convert-proton-to-protongpu="max-shared-mem-size=4096" -allocate-proton-shared-memory %s | FileCheck %s
// RUN: triton-opt --split-input-file -allocate-shared-memory -convert-proton-to-protongpu="max-shared-mem-size=4096" -allocate-proton-shared-memory -allocate-proton-shared-memory %s | FileCheck %s

#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
// CHECK: ttg.shared = 1664 : i32
// CHECK-SAME: \22reserved_bytes\22:1536
// CHECK-SAME: \22reservations\22:[{\22kind\22:\22profiling\22,{{[^{]*}}\22offset\22:1536,\22operation\22:\22ttg.local_alloc\22,\22size\22:128}]
// CHECK-SAME: \22reserved_bytes\22:1664
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: tt.func @allocate_aligned
  tt.func @allocate_aligned(%A : !tt.ptr<f16>) {
  %cst0 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  proton.record start "name0"
  %cst1 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %cst2 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  ttg.local_dealloc %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  ttg.local_dealloc %cst1 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  proton.record end "name0"
  ttg.local_dealloc %cst2 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK: ttg.local_alloc  {allocation.offset = 1536 : i32}
  tt.return
  }
}

// -----

#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0], CGALayout = [[1, 0]]}>
// CHECK: ttg.shared = 832 : i32
// CHECK-SAME: \22reserved_bytes\22:768
// CHECK-SAME: \22reservations\22:[{\22kind\22:\22profiling\22,{{[^{]*}}\22offset\22:768,\22operation\22:\22ttg.local_alloc\22,\22size\22:64}]
// CHECK-SAME: \22reserved_bytes\22:832
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 2 : i32} {
  // CHECK-LABEL: tt.func @allocate_aligned
  tt.func @allocate_aligned(%A : !tt.ptr<f16>) {
  %cst0 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  proton.record start "name0"
  %cst1 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %cst2 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  ttg.local_dealloc %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  ttg.local_dealloc %cst1 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  proton.record end "name0"
  ttg.local_dealloc %cst2 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK: ttg.local_alloc  {allocation.offset = 768 : i32}
  tt.return
  }
}

// -----

#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
// CHECK: ttg.shared = 64 : i32
// CHECK-SAME: \22reservations\22:[]
// CHECK-SAME: \22reserved_bytes\22:64
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: tt.func @no_proton
  tt.func @no_proton(%A : !tt.ptr<f16>) {
  %cst0 = ttg.local_alloc : () -> !ttg.memdesc<1x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  ttg.local_dealloc %cst0 : !ttg.memdesc<1x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK: ttg.local_alloc
  // CHECK-NOT: ttg.local_alloc
  tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// Preserve the one-byte kernel allocation and account for Proton alignment.
// CHECK: ttg.shared = 136 : i32
// CHECK-SAME: \22reserved_bytes\22:1
// CHECK-SAME: \22reservations\22:[{\22kind\22:\22alignment\22,{{[^{]*}}\22offset\22:1,\22operation\22:\22ttg.local_alloc\22,\22size\22:7},{\22kind\22:\22profiling\22,{{[^{]*}}\22offset\22:8,\22operation\22:\22ttg.local_alloc\22,\22size\22:128}]
// CHECK-SAME: \22reserved_bytes\22:136
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: tt.func @allocate_unaligned
  tt.func @allocate_unaligned() {
    %buffer = ttg.local_alloc : () -> !ttg.memdesc<1xi8, #shared, #ttg.shared_memory, mutable>
    proton.record start "scope"
    proton.record end "scope"
    ttg.local_dealloc %buffer : !ttg.memdesc<1xi8, #shared, #ttg.shared_memory, mutable>
    // CHECK: ttg.local_alloc {{.*}}allocation.offset = 8 : i32
    tt.return
  }
}
