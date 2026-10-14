// RUN: triton-opt -split-input-file --convert-triton-gpu-to-llvm=compute-capability=90 --verify-diagnostics %s
// RUN: triton-opt -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 --verify-diagnostics %s
// RUN: sed '/expected-error/d' %s | not triton-opt -split-input-file --convert-triton-gpu-to-llvm=compute-capability=80 2>&1 | FileCheck %s --check-prefix=SM80

// Strip SM90+ diagnostic annotations only from the SM80 input so diagnostic
// verification cannot report them as missing. Architecture rejection
// takes precedence over placement and motion diagnostics in every chunk.
// SM80: error: 'ttng.init_barrier' op explicit cluster initialization requires NVIDIA SM90+
// SM80-NOT: requires unconditional kernel-entry placement
// SM80-NOT: cannot safely hoist
// SM80-NOT: later-phase initialization
// SM80-NOT: reinitialization is unsupported

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  tt.func public @nested_if(%pred: i1) {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    scf.if %pred {
      // expected-error @+1 {{explicit cluster initialization requires unconditional kernel-entry placement after inlining}}
      ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    }
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  tt.func public @nested_loop() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c1 step %c1 : i32 {
      // expected-error @+1 {{explicit cluster initialization requires unconditional kernel-entry placement after inlining}}
      ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    }
    tt.return
  }
}

// -----

// The WS default region is not the unconditional kernel entry block.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  tt.func public @nested_ws_default() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      // expected-error @+1 {{explicit cluster initialization requires unconditional kernel-entry placement after inlining}}
      ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  tt.func public @nested_ws_partition() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize(%bar) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%local: !ttg.memdesc<1xi64, #shared, #smem, mutable>) num_warps(1) {
      // expected-error @+1 {{explicit cluster initialization requires unconditional kernel-entry placement after inlining}}
      ttng.init_barrier %local, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

// Intentionally omit --inline: a private helper cannot own the bootstrap.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  tt.func private @uninlined_helper(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) attributes {noinline = false} {
    // expected-error @+1 {{explicit cluster initialization requires unconditional kernel-entry placement after inlining}}
    ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
  tt.func public @call_uninlined_helper() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.call @uninlined_helper(%bar) : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

// Use distinct barriers so this diagnoses later-phase initialization, not a
// second initialization of the same descriptor.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  tt.func public @init_after_arrive() {
    %local = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %local, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.arrive_barrier %local, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-error @+1 {{explicit cluster initialization must precede effectful execution; later-phase initialization is unsupported}}
    ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  tt.func public @init_after_invalidate() {
    %local = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %local, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.inval_barrier %local : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-error @+1 {{explicit cluster initialization must precede effectful execution; later-phase initialization is unsupported}}
    ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// The index is in the entry block but depends on an effectful tt.load. The
// dependency diagnostic must win over the generic later-phase diagnostic.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  tt.func public @effectful_index_dependency(%ptr: !tt.ptr<i32>) {
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    %index = tt.load %ptr : !tt.ptr<i32>
    %bar = ttg.memdesc_index %bars[%index] : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-error @+1 {{explicit cluster initialization cannot safely hoist an effectful or non-entry dependency}}
    ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.cluster-dim-x" = 2 : i32} {
  tt.func public @reinitialize_same_descriptor() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-error @+1 {{explicit cluster barrier reinitialization is unsupported}}
    ttng.init_barrier %bar, 1 {bCluster = true} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
