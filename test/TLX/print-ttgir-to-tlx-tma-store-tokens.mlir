// RUN: triton-opt --tlx-print-ttgir-to-tlx %s -split-input-file | FileCheck %s

// A TMA store's async token only orders the store against its token wait, and
// tlx.async_descriptor_store_wait orders by count instead, so the token has no
// Python binding. Warp specialization threads these tokens through branches
// and loops, with ub.poison on the paths that issue no store: by the time the
// printer runs that is cf block arguments. Assigning them produced
// `None = ub.poison()` and branches yielding an unbound name or `None`, which
// the frontend rejects.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def guarded_store(
  // CHECK-NOT: ub.poison
  // CHECK: if
  // CHECK: tlx.async_descriptor_store(
  // CHECK-NOT: = None
  // CHECK: tlx.async_descriptor_store_wait(0)
  tt.func public @guarded_store(%desc: !tt.tensordesc<128x128xbf16, #shared>, %buf: !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, %c: i1, %x: i32) attributes {noinline = false} {
    %poison = ub.poison : !ttg.async.token
    cf.cond_br %c, ^bb1, ^bb2(%poison : !ttg.async.token)
  ^bb1:
    %t = ttng.async_tma_copy_local_to_global %desc[%x, %x] %buf : !tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.async.token
    cf.br ^bb2(%t : !ttg.async.token)
  ^bb2(%tok: !ttg.async.token):
    ttng.async_tma_store_token_wait %tok : !ttg.async.token
    tt.return
  }
}
