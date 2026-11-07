// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-clc-split -triton-nvidia-gpu-clc-hoist | FileCheck %s --check-prefix=HOIST
// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-clc-split -triton-nvidia-gpu-clc-hoist -triton-nvidia-gpu-clc-materialize | FileCheck %s --check-prefix=MAT

// Partial work-stealing: `clc_advance %steal` must issue no CLC request when
// %steal is false and report isValid = false, so a non-stealing CTA retires
// after its current tile without cancelling (and dropping) a pending cluster.

// HOIST-LABEL: @steal_loop_invariant
// The split keeps the predicate on the issue, and a predicate defined outside
// the loop does not limit the hoist to the top of the body.
// HOIST:       %[[STEAL:.*]] = arith.cmpi slt
// HOIST:       ^bb0(
// HOIST-NEXT:  %[[TOK:.*]] = ttng.clc_try_cancel_async %[[STEAL]] : !ttg.async.token
// HOIST-NEXT:  arith.addi
// HOIST-NEXT:  ttng.clc_read %[[TOK]]

// MAT-LABEL: @steal_loop_invariant
// MAT:       %[[STEAL:.*]] = arith.cmpi slt
// MAT:       %[[RESP:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
// MAT:       ttng.init_barrier
// MAT:       %[[BAR:.*]] = ttg.memdesc_index
// MAT:       scf.while
// MAT:       ^bb0(%{{.*}}: i1, %{{.*}}: i32, %[[PHASE:.*]]: i32):
// The expect and the request itself are both predicated on %steal...
// MAT-NEXT:  ttng.barrier_expect %[[BAR]], 16, %[[STEAL]]
// MAT-NEXT:  scf.if %[[STEAL]] {
// MAT-NEXT:    ttng.clc_try_cancel %[[RESP]], %[[BAR]]
// MAT-NEXT:  }
// ...as is the wait, and %steal masks the (otherwise stale) response.
// MAT:       ttng.wait_barrier %[[BAR]], %[[PHASE]], %[[STEAL]]
// MAT-NEXT:  %[[RES:.*]] = ttng.clc_load_result %[[RESP]]
// MAT-NEXT:  %[[CANCELED:.*]] = ttng.clc_is_canceled %[[RES]]
// MAT-NEXT:  %[[VALID:.*]] = arith.andi %[[CANCELED]], %[[STEAL]] : i1
// MAT:       scf.yield %[[VALID]]

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @steal_loop_invariant(%n: i32) {
    %pid = tt.get_program_id x : i32
    %steal = arith.cmpi slt, %pid, %n : i32
    %true = arith.constant true
    %r:2 = scf.while (%v = %true, %x = %pid) : (i1, i32) -> (i1, i32) {
      scf.condition(%v) %v, %x : i1, i32
    } do {
    ^bb0(%v: i1, %x: i32):
      %c = arith.addi %x, %x : i32
      %valid, %nx, %ny, %nz = ttng.clc_advance %steal : i1, i32, i32, i32
      scf.yield %valid, %nx : i1, i32
    }
    tt.return
  }
}

// -----

// A predicate computed inside the loop body bounds the hoist: the issue moves
// up to just after the predicate's definition, not to the top of the body.

// HOIST-LABEL: @steal_in_body
// HOIST:       ^bb0(
// HOIST-NEXT:  arith.addi
// HOIST-NEXT:  %[[STEAL:.*]] = arith.cmpi slt
// HOIST-NEXT:  ttng.clc_try_cancel_async %[[STEAL]]
// HOIST-NEXT:  arith.muli
// HOIST-NEXT:  ttng.clc_read

// MAT-LABEL: @steal_in_body
// MAT:       %[[STEAL:.*]] = arith.cmpi slt
// MAT-NEXT:  ttng.barrier_expect %{{.*}}, 16, %[[STEAL]]
// MAT-NEXT:  scf.if %[[STEAL]] {

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @steal_in_body(%n: i32) {
    %pid = tt.get_program_id x : i32
    %true = arith.constant true
    %r:2 = scf.while (%v = %true, %x = %pid) : (i1, i32) -> (i1, i32) {
      scf.condition(%v) %v, %x : i1, i32
    } do {
    ^bb0(%v: i1, %x: i32):
      %c = arith.addi %x, %x : i32
      %steal = arith.cmpi slt, %c, %n : i32
      %d = arith.muli %c, %c : i32
      %valid, %nx, %ny, %nz = ttng.clc_advance %steal : i1, i32, i32, i32
      scf.yield %valid, %nx : i1, i32
    }
    tt.return
  }
}

// -----

// Explicit 2-CTA cluster: only the leader issues, and only when %steal holds;
// the leader's empty-barrier wait is predicated the same way. The release
// arrival on the leader's empty barrier is predicated on the masked isValid,
// so a non-stealing cluster neither waits nor arrives.

// MAT-LABEL: @steal_cluster
// MAT:       %[[STEAL:.*]] = arith.cmpi slt
// MAT:       scf.while
// MAT:       %[[RANK:.*]] = nvg.cluster_id
// MAT:       %[[LEADER:.*]] = arith.cmpi eq, %[[RANK]]
// MAT:       %[[WAITP:.*]] = arith.andi %[[LEADER]], %[[STEAL]] : i1
// MAT-NEXT:  ttng.wait_barrier %{{.*}}, %{{.*}}, %[[WAITP]]
// MAT-NEXT:  ttng.barrier_expect %{{.*}}, 16, %[[STEAL]]
// MAT-NEXT:  scf.if %[[STEAL]] {
// MAT-NEXT:    ttng.clc_try_cancel
// MAT-NEXT:  }
// MAT-NEXT:  ttng.wait_barrier %{{.*}}, %{{.*}}, %[[STEAL]]
// MAT:       %[[CANCELED:.*]] = ttng.clc_is_canceled
// MAT-NEXT:  %[[VALID:.*]] = arith.andi %[[CANCELED]], %[[STEAL]] : i1
// MAT:       ttng.arrive_barrier %{{.*}}, 1, %[[VALID]]
// MAT:       scf.yield %[[VALID]]

module attributes {"ttg.cluster-dim-x" = 2 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @steal_cluster(%n: i32) {
    %pid = tt.get_program_id x : i32
    %c2 = arith.constant 2 : i32
    %cid = arith.divsi %pid, %c2 : i32
    %steal = arith.cmpi slt, %cid, %n : i32
    %true = arith.constant true
    %r:2 = scf.while (%v = %true, %x = %pid) : (i1, i32) -> (i1, i32) {
      scf.condition(%v) %v, %x : i1, i32
    } do {
    ^bb0(%v: i1, %x: i32):
      %valid, %nx, %ny, %nz = ttng.clc_advance %steal : i1, i32, i32, i32
      scf.yield %valid, %nx : i1, i32
    }
    tt.return
  }
}

// -----

// Without a predicate the lowering is unchanged: no scf.if, unpredicated wait.

// MAT-LABEL: @no_steal_predicate
// MAT:       ttng.barrier_expect %{{.*}}, 16, %true
// MAT-NEXT:  ttng.clc_try_cancel
// MAT-NEXT:  ttng.wait_barrier %{{[0-9]+}}, %{{[a-z0-9]+}} :
// MAT-NEXT:  ttng.clc_load_result
// MAT-NEXT:  %[[CANCELED:.*]] = ttng.clc_is_canceled
// MAT-NOT:   arith.andi
// MAT:       scf.yield %[[CANCELED]]

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_steal_predicate(%n: i32) {
    %pid = tt.get_program_id x : i32
    %true = arith.constant true
    %r:2 = scf.while (%v = %true, %x = %pid) : (i1, i32) -> (i1, i32) {
      scf.condition(%v) %v, %x : i1, i32
    } do {
    ^bb0(%v: i1, %x: i32):
      %valid, %nx, %ny, %nz = ttng.clc_advance : i1, i32, i32, i32
      scf.yield %valid, %nx : i1, i32
    }
    tt.return
  }
}
