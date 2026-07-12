// RUN: triton-opt %s -allow-unregistered-dialect -test-print-allocation \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: TRITON_USE_META_WS=1 triton-opt %s -allow-unregistered-dialect \
// RUN:   -test-print-allocation -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=META

#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // Without MetaWS, the join between sequential warp-specialize operations
  // permits both their explicit buffers and capture scratch to be reused.
  // DEFAULT-LABEL: sequential_warp_specialize_liveness
  // DEFAULT: offset = 0, size = 16
  // DEFAULT: scratch offset = 16, size = 12
  // DEFAULT: offset = 0, size = 16
  // DEFAULT: scratch offset = 16, size = 12
  // DEFAULT: size = 29

  // MetaWS retains its conservative function-entry lifetime workaround.
  // META-LABEL: sequential_warp_specialize_liveness
  // META: offset = 0, size = 16
  // META: scratch offset = 32, size = 12
  // META: offset = 16, size = 16
  // META: scratch offset = 48, size = 12
  // META: size = 61
  tt.func @sequential_warp_specialize_liveness() {
    %first = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    ttg.warp_specialize(%first)
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2xi64, #shared, #smem, mutable>) num_warps(1) {
      "use"(%arg0) : (!ttg.memdesc<2xi64, #shared, #smem, mutable>) -> ()
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #shared, #smem, mutable>) -> ()

    %second = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    ttg.warp_specialize(%second)
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2xi64, #shared, #smem, mutable>) num_warps(1) {
      "use"(%arg0) : (!ttg.memdesc<2xi64, #shared, #smem, mutable>) -> ()
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
