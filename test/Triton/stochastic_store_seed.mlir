// RUN: triton-opt %s --triton-materialize-stochastic-rounding='mode=enabled site=output compute-capability=100 ptx-version=87 rounds=10' | FileCheck %s

// The logical index is computed before a reduction loop that does not use the
// seed. The seed load and the entropy follow the loop.

// CHECK-LABEL: tt.func public @reduction_then_store
// CHECK-NOT: tt.load %arg0
// CHECK: scf.for
// CHECK: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: tt.load %arg0 : !tt.ptr<i64>
// CHECK: tt.mulhiui
// CHECK: tt.store
module {
  tt.func public @reduction_then_store(%seed: !tt.ptr<i64>, %base_seed: i64, %invocation: i64, %output: !tt.ptr<bf16>, %input: !tt.ptr<f32>, %n: i32) {
    %pid = tt.get_program_id x : i32
    %p64 = arith.extui %pid : i32 to i64
    %c128 = arith.constant 128 : i64
    %block = arith.muli %p64, %c128 : i64
    %splat = tt.splat %block : i64 -> tensor<128xi64>
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %r64 = arith.extui %range : tensor<128xi32> to tensor<128xi64>
    %index = arith.addi %splat, %r64 : tensor<128xi64>
    %in = tt.splat %input : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %inp = tt.addptr %in, %range : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<128xf32>
    %acc = scf.for %i = %c0 to %n step %c1 iter_args(%a = %zero) -> (tensor<128xf32>) : i32 {
      %v = tt.load %inp : tensor<128x!tt.ptr<f32>>
      %s = arith.addf %a, %v : tensor<128xf32>
      scf.yield %s : tensor<128xf32>
    }
    %base = tt.splat %output : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %ptr = tt.addptr %base, %index : tensor<128x!tt.ptr<bf16>>, tensor<128xi64>
    tt.sr_store %ptr, %acc, %index {site = "output"} : (tensor<128x!tt.ptr<bf16>>, tensor<128xf32>, tensor<128xi64>) -> ()
    tt.return
  }
}
