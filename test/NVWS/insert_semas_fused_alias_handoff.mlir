// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s --check-prefix=SEMA
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-assign-stage-phase -cse | FileCheck %s --check-prefix=ASP

// Two exact-alias epilogue members share one depth-2 physical allocation.
// The first member uses slot 0 and the second uses slot 1, so each
// read-to-next-write release must target the successor slot rather than the
// source read's slot.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SEMA-LABEL: @fused_alias_depth_two
  // SEMA: [[BASE:%.*]] = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32}
  // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] true
  // SEMA: [[FULL0:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] false
  // SEMA: [[EMPTY1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] false
  // SEMA: [[FULL1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] false
  // SEMA: scf.for
  // SEMA: [[W0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: [[W0_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[W0_ZERO]]]
  // SEMA: [[W0_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: nvws.semaphore.release [[FULL0]][[[W0_REL_ZERO]]], [[W0_TOK]]
  // SEMA: [[R0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
  // SEMA: [[R0_TOK:%.*]] = nvws.semaphore.acquire [[FULL0]][[[R0_ZERO]]]
  // SEMA: [[TO_M1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
  // SEMA: nvws.semaphore.release [[EMPTY1]][[[TO_M1]]], [[R0_TOK]]
  // SEMA: [[W1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: [[W1_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY1]][[[W1_ZERO]]]
  // SEMA: [[W1_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: nvws.semaphore.release [[FULL1]][[[W1_REL_ZERO]]], [[W1_TOK]]
  // SEMA: [[R1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
  // SEMA: [[R1_TOK:%.*]] = nvws.semaphore.acquire [[FULL1]][[[R1_ZERO]]]
  // SEMA: [[TO_NEXT_M0:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
  // SEMA: nvws.semaphore.release [[ENTRY]][[[TO_NEXT_M0]]], [[R1_TOK]]

  // ASP-LABEL: @fused_alias_depth_two
  // ASP: [[ENTRY:%.*]] = nvws.semaphore.create
  // ASP: [[FULL0:%.*]] = nvws.semaphore.create
  // ASP: [[EMPTY1:%.*]] = nvws.semaphore.create
  // ASP: [[FULL1:%.*]] = nvws.semaphore.create
  // ASP: scf.for {{.*}} iter_args([[CURSOR:%.*]] = {{%.*}}
  // ASP: [[SLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: arith.shli {{%.*}}, [[SLOT0]] {ttg.partition = array<i32: 4>} : i32
  // ASP: [[W0_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[SLOT0]], {{%.*}}]
  // ASP: nvws.semaphore.release [[FULL0]][[[SLOT0]]], [[W0_TOK]]
  // ASP: [[R0_TOK:%.*]] = nvws.semaphore.acquire [[FULL0]][[[SLOT0]], {{%.*}}]
  // ASP: [[TO_M1_RAW:%.*]] = arith.addi [[SLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[TO_M1_REM:%.*]] = arith.remsi [[TO_M1_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[TO_M1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_M1_REM]] {ttg.partition = array<i32: 2>} : i32
  // ASP: nvws.semaphore.release [[EMPTY1]][[[TO_M1]]], [[R0_TOK]]
  // ASP: [[NEXT_RAW:%.*]] = arith.addi [[SLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: [[SLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[NEXT_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: arith.shli {{%.*}}, [[SLOT1]] {ttg.partition = array<i32: 4>} : i32
  // ASP: [[W1_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY1]][[[SLOT1]], {{%.*}}]
  // ASP: nvws.semaphore.release [[FULL1]][[[SLOT1]]], [[W1_TOK]]
  // ASP: [[R1_TOK:%.*]] = nvws.semaphore.acquire [[FULL1]][[[SLOT1]], {{%.*}}]
  // ASP: [[TO_M0_RAW:%.*]] = arith.addi [[SLOT1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[TO_M0_REM:%.*]] = arith.remsi [[TO_M0_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[TO_M0:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_M0_REM]] {ttg.partition = array<i32: 2>} : i32
  // ASP: nvws.semaphore.release [[ENTRY]][[[TO_M0]]], [[R1_TOK]]
  // ASP: scf.yield {{.*}} [[SLOT1]],
  tt.func @fused_alias_depth_two(%lb: i32, %ub: i32, %step: i32) {
    %m0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %m1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>

    scf.for %iv = %lb to %ub step %step : i32 {
      ttg.local_store %v0, %m0 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %r0 = ttg.local_load %m0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "consume0"(%r0) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
      ttg.local_store %v1, %m1 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %r1 = ttg.local_load %m1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "consume1"(%r1) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
