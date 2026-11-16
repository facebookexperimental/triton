// RUN: triton-opt --tlx-print-ttgir-to-tlx %s -split-input-file | FileCheck %s

// A retyping memdesc_reinterpret prints as tlx.local_alloc(..., reuse=base),
// and TLXRewriteLocalAlias requires `base` to be an allocation. Inside a
// warp_specialize partition the allocation is only a capture, so the alias
// failed the pass. AutoWS emits exactly this to alias P over QK in tensor
// memory. Partition reinterprets of a captured allocation are hoisted ahead of
// tlx.async_tasks(), one per allocation, element type and shape, and the partitions name
// the hoisted alias, which TLX captures like any other live-in.

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_p = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def partition_reinterpret(
  // CHECK: qk = tlx.local_alloc((128, 128), tl.float32, 1, tlx.storage_kind.tmem)
  // CHECK-NEXT: qk_as_bf16_1_128_256 = tlx.local_alloc((128, 256), tl.bfloat16, 1, tlx.storage_kind.tmem, reuse=qk)
  // CHECK-NOT: reuse=
  // CHECK: with tlx.async_tasks():
  // CHECK: with tlx.async_task(num_warps=4
  // CHECK-NEXT: {{[a-z_0-9]+}} = qk_as_bf16_1_128_256
  // CHECK: with tlx.async_task(num_warps=4
  // CHECK-NEXT: {{[a-z_0-9]+}} = qk_as_bf16_1_128_256
  tt.func public @partition_reinterpret() attributes {noinline = false} {
    %qk = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc("qk")
    ttg.warp_specialize(%qk)
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
      %p = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x256xbf16, #tmem_p, #ttng.tensor_memory, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
      %p = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x256xbf16, #tmem_p, #ttng.tensor_memory, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}

// -----

// Views that differ only in encoding print the same alias line, so they share
// one hoisted name instead of defining it twice.

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_p = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#tmem_p2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def encoding_only_difference(
  // CHECK: qk_as_bf16_1_128_256 = tlx.local_alloc(
  // CHECK-NOT: qk_as_bf16_1_128_256 = tlx.local_alloc(
  // CHECK: with tlx.async_tasks():
  // CHECK: {{[a-z_0-9]+}} = qk_as_bf16_1_128_256
  // CHECK: {{[a-z_0-9]+}} = qk_as_bf16_1_128_256
  tt.func public @encoding_only_difference() attributes {noinline = false} {
    %qk = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc("qk")
    ttg.warp_specialize(%qk)
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
      %p = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x256xbf16, #tmem_p, #ttng.tensor_memory, mutable>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(4) {
      %p = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x256xbf16, #tmem_p2, #ttng.tensor_memory, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
    tt.return
  }
}
