// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-prepare-consan-captures="target=nvidia" -tritoninstrument-concurrency-sanitizer | FileCheck %s
// RUN: env -u TRITON_ALLOW_NPOT not triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-prepare-consan-captures="target=nvidia" -tritoninstrument-concurrency-sanitizer -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF

// There is no default tcgen05.ld/st layout for an NPOT TMEM tile, so ConSan
// does not poison a TMEM pow2 alloc shape. With the flag off the verifier
// rejects it.

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
// CHECK-LABEL: @tmem_pow2_alloc
// CHECK: ttng.tmem_alloc
// CHECK-NOT: ttng.tmem_store
// CHECK: tt.return
// OFF: shape must have power-of-2 and non-zero dimensions; got 128, 96
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @tmem_pow2_alloc() {
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x96xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

// A tensor cannot hold the NPOT tile of a compact shared allocation, so
// ConSan does not poison its pages; the power-of-two control's pages are.
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: @swizzled_pages
// CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<2x64x96xf16, #shared, #smem, mutable>
// CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<2x64x128xf16, #shared, #smem, mutable>
// CHECK-COUNT-2: ttg.local_store {{.*}} -> !ttg.memdesc<64x128xf16
// CHECK-NOT: ttg.local_store
// CHECK: tt.return
// OFF: shape must have power-of-2 and non-zero dimensions; got 2, 64, 96
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @swizzled_pages() {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x96xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<2x64x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Power-of-two control: the TMEM allocation is still poisoned.
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
// CHECK-LABEL: @tmem_pot
// CHECK: %[[T:.*]] = ttng.tmem_alloc
// CHECK: ttng.tmem_store %{{.*}}, %[[T]]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @tmem_pot() {
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
