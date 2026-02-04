// RUN: triton-opt %s -triton-nvidia-gpu-select-to-branch | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @test_select_to_branch_with_ballot
  tt.func @test_select_to_branch_with_ballot(
      %memdesc: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>,
      %pred: i1,
      %alpha: tensor<128x64xf32, #blocked>
  ) {
    // CHECK: ttng.vote_ballot_sync
    // CHECK: tt.splat
    // CHECK: arith.cmpi
    // CHECK: ttng.if_from_where %{{.*}} {
    // CHECK:   ttng.tmem_load
    // CHECK:   arith.mulf
    // CHECK:   ttng.tmem_store
    // CHECK:   ttng.if_from_where_yield
    // CHECK: }
    // CHECK-NOT: arith.select

    %true = arith.constant true
    %mask = arith.constant 0xFFFFFFFF : i32
    %c0 = arith.constant 0 : i32

    // vote_ballot_sync returns i32
    %ballot = ttng.vote_ballot_sync %mask, %pred : i1 -> i32
    // splat to tensor
    %ballot_splat = tt.splat %ballot : i32 -> tensor<128x64xi32, #blocked>
    // compare with 0 to get bool tensor
    %c0_splat = tt.splat %c0 : i32 -> tensor<128x64xi32, #blocked>
    %should_rescale = arith.cmpi ne, %ballot_splat, %c0_splat : tensor<128x64xi32, #blocked>

    %acc = ttng.tmem_load %memdesc : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    %scaled = arith.mulf %acc, %alpha : tensor<128x64xf32, #blocked>
    %result = arith.select %should_rescale, %scaled, %acc : tensor<128x64xi1, #blocked>, tensor<128x64xf32, #blocked>
    ttng.tmem_store %result, %memdesc, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

    tt.return
  }
}
