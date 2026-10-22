// RUN: split-file %s %t
// RUN: triton-opt %t/dynamic.mlir --triton-nvidia-gpu-promote-mbarrier-to-named-barrier --triton-nvidia-gpu-verify-named-barrier-counts --canonicalize --allocate-shared-memory-nv --convert-scf-to-cf --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-nv-gpu-to-llvm --convert-warp-specialize-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --canonicalize --reconcile-unrealized-casts > %t/dynamic.llir
// RUN: FileCheck %s --check-prefix=LLVM < %t/dynamic.llir
// RUN: mlir-translate %t/dynamic.llir --mlir-to-llvmir | opt -O3 -S | llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 -mattr=+ptx87 | FileCheck %s --check-prefix=DYNAMIC --implicit-check-not=mbarrier
// RUN: triton-opt %t/empty.mlir --triton-nvidia-gpu-promote-mbarrier-to-named-barrier --triton-nvidia-gpu-verify-named-barrier-counts --canonicalize --allocate-shared-memory-nv --convert-scf-to-cf --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-nv-gpu-to-llvm --convert-warp-specialize-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --canonicalize --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 -mattr=+ptx87 | FileCheck %s --check-prefix=EMPTY --implicit-check-not=mbarrier
// RUN: triton-opt %t/one.mlir --triton-nvidia-gpu-promote-mbarrier-to-named-barrier --triton-nvidia-gpu-verify-named-barrier-counts --canonicalize --allocate-shared-memory-nv --convert-scf-to-cf --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-nv-gpu-to-llvm --convert-warp-specialize-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --canonicalize --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 -mattr=+ptx87 | FileCheck %s --check-prefix=ONE --implicit-check-not=mbarrier

// LLVM-LABEL: llvm.func @peel_ptx_dynamic
// LLVM: %[[COUNT:.*]] = llvm.mlir.constant(64 : i32)
// LLVM: %[[BAR:.*]] = llvm.mlir.constant(4 : i32)
// LLVM: llvm.call_intrinsic "llvm.nvvm.barrier.cta.arrive.aligned.count"(%[[BAR]], %[[COUNT]])
// LLVM: %[[PRED:.*]] = llvm.icmp "slt" %arg0, %arg1
// LLVM: llvm.cond_br %[[PRED]], ^[[FIRST:bb[0-9]+]], ^[[JOIN:bb[0-9]+]]
// LLVM: ^[[FIRST]]:
// LLVM-NEXT: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.aligned.count"(%[[BAR]], %[[COUNT]])
// LLVM-NEXT: llvm.br ^[[JOIN]]
// LLVM: ^[[JOIN]]:
// LLVM: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.aligned.count"(%[[BAR]], %[[COUNT]])
// DYNAMIC-LABEL: .visible .entry peel_ptx_dynamic(
// DYNAMIC: setp.ge.s32 %[[SKIP_PRED:p[0-9]+]],
// DYNAMIC: @%[[SKIP_PRED]] bra [[SKIP:\$L__BB[0-9_]+]];
// DYNAMIC: bar.sync 4, 64;
// DYNAMIC: [[SKIP]]:
// DYNAMIC: [[LOOP:\$L__BB[0-9_]+]]:
// DYNAMIC: bar.sync 4, 64;
// DYNAMIC: @{{.*}} bra [[LOOP]];
// DYNAMIC: bar.arrive 4, 64;
// DYNAMIC-NOT: mbarrier
// EMPTY-LABEL: .visible .entry peel_ptx_empty(
// EMPTY-NOT: bar.arrive
// EMPTY-NOT: bar.sync {{.*}}4, 64
// EMPTY-NOT: mbarrier
// ONE-LABEL: .visible .entry peel_ptx_one(
// ONE-DAG: bar.arrive 4, 64
// ONE-DAG: bar.sync 4, 64
// ONE-NOT: bar.sync {{.*}}4, 64
// ONE-NOT: mbarrier

//--- dynamic.mlir
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @peel_ptx_dynamic(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.addi %begin, %c0 : i32
      %upper = arith.addi %end, %c0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

//--- empty.mlir
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @peel_ptx_empty(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant 0 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant 0 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}

//--- one.mlir
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 6 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @peel_ptx_one(%lb: i32, %ub: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %storage, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.warp_specialize(%storage, %lb, %ub) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 5>}
    default {
      ttg.warp_yield
    }
    partition0(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant 1 : i32
      %result = scf.for %i = %lower to %upper step %step iter_args(%counter = %counter0) -> (i64) : i32 {
        ttng.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %counter1 : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    }
    partition1(%b: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %begin: i32, %end: i32) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %step = arith.constant 3 : i32
      %counter0 = arith.constant 0 : i64
      %counter1 = arith.constant 1 : i64
      %lower = arith.constant 0 : i32
      %upper = arith.constant 1 : i32
      %phase = arith.constant 0 : i32
      %counter_init = arith.constant 1 : i64
      %increment = arith.constant 1 : i64
      %pred = arith.cmpi slt, %lower, %upper : i32
      %short = arith.subi %upper, %step : i32
      ttng.wait_barrier %b, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
      %result = scf.for %i = %lower to %short step %step iter_args(%counter = %counter_init) -> (i64) : i32 {
        %parity = arith.andi %counter, %counter1 : i64
        %bit = arith.trunci %parity : i64 to i1
        %loop_phase = arith.extui %bit : i1 to i32
        ttng.wait_barrier %b, %loop_phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
        %next = arith.addi %counter, %increment : i64
        scf.yield %next : i64
      }
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #barrier, #smem, mutable>, i32, i32) -> ()
    tt.return
  }
}
