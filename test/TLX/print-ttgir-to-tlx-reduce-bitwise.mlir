// RUN: triton-opt --tlx-print-ttgir-to-tlx %s -split-input-file | FileCheck %s

// Bitwise reduce combiners. AutoWS Flash Attention decides whether to rescale
// the accumulator with an OR vote over the rows, and the combiner detector
// only knew max/min/sum, so the vote fell back to a bare tl.reduce missing its
// combine_fn. The structured printer and the CF printer each detect combiners,
// and the printer sees CF by the time a real kernel is dumped, so both run.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def or_xor_reduce(
  // CHECK-DAG: tl.reduce_or(arg0, 0)
  // CHECK-DAG: tl.xor_sum(arg1, 0)
  // CHECK-NOT: tl.reduce(
  tt.func public @or_xor_reduce(%v: tensor<128xi1, #blocked>, %x: tensor<128xi32, #blocked>) attributes {noinline = false} {
    %any = "tt.reduce"(%v) <{axis = 0 : i32}> ({
    ^bb0(%a: i1, %b: i1):
      %o = arith.ori %a, %b : i1
      tt.reduce.return %o : i1
    }) : (tensor<128xi1, #blocked>) -> i1
    %par = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%c: i32, %d: i32):
      %e = arith.xori %c, %d : i32
      tt.reduce.return %e : i32
    }) : (tensor<128xi32, #blocked>) -> i32
    tt.return
  }

  // CHECK-LABEL: def or_reduce_in_cf(
  // CHECK: tl.reduce_or(arg0, 0)
  // CHECK-NOT: tl.reduce(
  tt.func public @or_reduce_in_cf(%v: tensor<128xi1, #blocked>, %c: i1) attributes {noinline = false} {
    cf.cond_br %c, ^bb1, ^bb2
  ^bb1:
    %any = "tt.reduce"(%v) <{axis = 0 : i32}> ({
    ^bb0(%a: i1, %b: i1):
      %o = arith.ori %a, %b : i1
      tt.reduce.return %o : i1
    }) : (tensor<128xi1, #blocked>) -> i1
    cf.br ^bb2
  ^bb2:
    tt.return
  }
}
