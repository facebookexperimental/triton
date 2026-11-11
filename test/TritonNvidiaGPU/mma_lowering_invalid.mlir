// RUN: triton-opt %s -split-input-file --triton-nvidia-mma-lowering --verify-diagnostics

#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared_scale = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>
#shared_barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, ctaMode = twocta_rhs>
#smem = #ttg.shared_memory

module attributes {tlx.enable_paired_cta_mma = true, "ttg.cluster-dim-x" = 2 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @two_cta_m64_mismatched_publication_barriers(
      %a: !ttg.memdesc<64x128xf8E4M3FN, #shared_a, #smem>,
      %b: !ttg.memdesc<128x64xf8E4M3FN, #shared_b, #smem>,
      %acc: !ttg.memdesc<64x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      %a_scale: !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>,
      %b_scale: !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>,
      %arrive_barrier: !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>,
      %wait_barrier: !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>,
      %mma_barrier: !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>) {
    %true = arith.constant true
    %false = arith.constant false
    %phase = arith.constant 0 : i32
    %remote_barrier = ttng.map_to_remote_buffer %arrive_barrier, %phase : !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #shared_barrier, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %remote_barrier, 1 : !ttg.memdesc<1xi64, #shared_barrier, #ttng.shared_cluster_memory, mutable>
    ttng.wait_barrier %wait_barrier, %phase : !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>
    // expected-error @below {{two-CTA blockM=64 scaled MMA requires a matched inter-CTA arrive/wait barrier after publishing its scales}}
    ttng.tc_gen5_mma_scaled %a, %b, %acc, %a_scale, %b_scale, %false, %true lhs = e4m3 rhs = e4m3, %mma_barrier[%true] {is_async, two_ctas} : !ttg.memdesc<64x128xf8E4M3FN, #shared_a, #smem>, !ttg.memdesc<128x64xf8E4M3FN, #shared_b, #smem>, !ttg.memdesc<64x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>, !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>, !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>
    tt.return
  }

  tt.func public @two_cta_m64_false_publication_wait(
      %a: !ttg.memdesc<64x128xf8E4M3FN, #shared_a, #smem>,
      %b: !ttg.memdesc<128x64xf8E4M3FN, #shared_b, #smem>,
      %acc: !ttg.memdesc<64x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      %a_scale: !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>,
      %b_scale: !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>,
      %publication_barrier: !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>,
      %mma_barrier: !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>) {
    %true = arith.constant true
    %false = arith.constant false
    %phase = arith.constant 0 : i32
    ttng.cluster_barrier
    %remote_barrier = ttng.map_to_remote_buffer %publication_barrier, %phase : !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #shared_barrier, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %remote_barrier, 1 : !ttg.memdesc<1xi64, #shared_barrier, #ttng.shared_cluster_memory, mutable>
    ttng.wait_barrier %publication_barrier, %phase, %false : !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>
    // expected-error @below {{two-CTA blockM=64 scaled MMA requires a matched inter-CTA arrive/wait barrier after publishing its scales}}
    ttng.tc_gen5_mma_scaled %a, %b, %acc, %a_scale, %b_scale, %false, %true lhs = e4m3 rhs = e4m3, %mma_barrier[%true] {is_async, two_ctas} : !ttg.memdesc<64x128xf8E4M3FN, #shared_a, #smem>, !ttg.memdesc<128x64xf8E4M3FN, #shared_b, #smem>, !ttg.memdesc<64x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>, !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>, !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>
    tt.return
  }

  tt.func public @two_cta_m64_unmatched_cluster_wait_is_terminal(
      %a: !ttg.memdesc<64x128xf8E4M3FN, #shared_a, #smem>,
      %b: !ttg.memdesc<128x64xf8E4M3FN, #shared_b, #smem>,
      %acc: !ttg.memdesc<64x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      %a_scale: !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>,
      %b_scale: !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>,
      %mma_barrier: !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>) {
    %true = arith.constant true
    %false = arith.constant false
    ttng.cluster_arrive
    ttng.cluster_wait
    ttng.cluster_wait
    // expected-error @below {{two-CTA blockM=64 scaled MMA requires a matched inter-CTA arrive/wait barrier after publishing its scales}}
    ttng.tc_gen5_mma_scaled %a, %b, %acc, %a_scale, %b_scale, %false, %true lhs = e4m3 rhs = e4m3, %mma_barrier[%true] {is_async, two_ctas} : !ttg.memdesc<64x128xf8E4M3FN, #shared_a, #smem>, !ttg.memdesc<128x64xf8E4M3FN, #shared_b, #smem>, !ttg.memdesc<64x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>, !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>, !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>
    tt.return
  }

  tt.func public @two_cta_m64_stale_mbarrier_arrive_is_not_reused(
      %a: !ttg.memdesc<64x128xf8E4M3FN, #shared_a, #smem>,
      %b: !ttg.memdesc<128x64xf8E4M3FN, #shared_b, #smem>,
      %acc: !ttg.memdesc<64x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      %a_scale: !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>,
      %b_scale: !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>,
      %publication_barrier: !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>,
      %mma_barrier: !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>) {
    %true = arith.constant true
    %false = arith.constant false
    %phase = arith.constant 0 : i32
    %remote_barrier = ttng.map_to_remote_buffer %publication_barrier, %phase : !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #shared_barrier, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %remote_barrier, 1 : !ttg.memdesc<1xi64, #shared_barrier, #ttng.shared_cluster_memory, mutable>
    ttng.wait_barrier %publication_barrier, %phase : !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>
    ttng.wait_barrier %publication_barrier, %phase : !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>
    // expected-error @below {{two-CTA blockM=64 scaled MMA requires a matched inter-CTA arrive/wait barrier after publishing its scales}}
    ttng.tc_gen5_mma_scaled %a, %b, %acc, %a_scale, %b_scale, %false, %true lhs = e4m3 rhs = e4m3, %mma_barrier[%true] {is_async, two_ctas} : !ttg.memdesc<64x128xf8E4M3FN, #shared_a, #smem>, !ttg.memdesc<128x64xf8E4M3FN, #shared_b, #smem>, !ttg.memdesc<64x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>, !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>, !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>
    tt.return
  }
}

// -----

#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared_scale = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>
#shared_barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1, ctaMode = twocta_rhs>
#smem = #ttg.shared_memory

module attributes {tlx.enable_paired_cta_mma = true, "ttg.cluster-dim-x" = 2 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  tt.func public @two_cta_m64_scale_view_after_publication_barrier(
      %a: !ttg.memdesc<64x128xf8E4M3FN, #shared_a, #smem>,
      %b: !ttg.memdesc<128x64xf8E4M3FN, #shared_b, #smem>,
      %acc: !ttg.memdesc<64x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      %a_scale_storage: !ttg.memdesc<1x1x1x1x2x256xi8, #shared_scale, #smem>,
      %b_scale_storage: !ttg.memdesc<1x1x1x1x2x256xi8, #shared_scale, #smem>,
      %publication_barrier: !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>,
      %mma_barrier: !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>) {
    %true = arith.constant true
    %false = arith.constant false
    %phase = arith.constant 0 : i32
    %remote_barrier = ttng.map_to_remote_buffer %publication_barrier, %phase : !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #shared_barrier, #ttng.shared_cluster_memory, mutable>
    ttng.arrive_barrier %remote_barrier, 1 : !ttg.memdesc<1xi64, #shared_barrier, #ttng.shared_cluster_memory, mutable>
    ttng.wait_barrier %publication_barrier, %phase : !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>
    %a_scale = ttg.memdesc_index %a_scale_storage[%phase] : !ttg.memdesc<1x1x1x1x2x256xi8, #shared_scale, #smem> -> !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>
    %b_scale = ttg.memdesc_index %b_scale_storage[%phase] : !ttg.memdesc<1x1x1x1x2x256xi8, #shared_scale, #smem> -> !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>
    // expected-error @below {{two-CTA blockM=64 shared-memory scale operands must be defined before their publication barrier}}
    ttng.tc_gen5_mma_scaled %a, %b, %acc, %a_scale, %b_scale, %false, %true lhs = e4m3 rhs = e4m3, %mma_barrier[%true] {is_async, two_ctas} : !ttg.memdesc<64x128xf8E4M3FN, #shared_a, #smem>, !ttg.memdesc<128x64xf8E4M3FN, #shared_b, #smem>, !ttg.memdesc<64x128xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>, !ttg.memdesc<1x1x1x2x256xi8, #shared_scale, #smem>, !ttg.memdesc<1xi64, #shared_barrier, #smem, mutable>
    tt.return
  }
}
