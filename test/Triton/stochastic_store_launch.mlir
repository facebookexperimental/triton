// RUN: triton-opt %s --split-input-file --triton-materialize-stochastic-rounding='mode=enabled site=output compute-capability=100 ptx-version=87 rounds=10' --verify-diagnostics | FileCheck %s

// The pointer and logical index use different strides, so only the
// launch-aware proof can accept the store, and it needs the module grid.

// CHECK-LABEL: tt.func public @sharded_with_grid
// CHECK-NOT: tt.sr_store
// CHECK: tt.store
module attributes {tt.sr_grid = 4 : i32} {
  tt.func public @sharded_with_grid(%seed: !tt.ptr<i64>, %base_seed: i64, %invocation: i64, %output: !tt.ptr<bf16>, %x: tensor<128xf32>) {
    %pid = tt.get_program_id x : i32
    %p = arith.extsi %pid : i32 to i64
    %c128 = arith.constant 128 : i64
    %c512 = arith.constant 512 : i64
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %r = arith.extsi %range : tensor<128xi32> to tensor<128xi64>
    %local = arith.muli %p, %c128 : i64
    %local_splat = tt.splat %local : i64 -> tensor<128xi64>
    %offset = arith.addi %local_splat, %r : tensor<128xi64>
    %global = arith.muli %p, %c512 : i64
    %shifted = arith.addi %global, %c128 : i64
    %global_splat = tt.splat %shifted : i64 -> tensor<128xi64>
    %index = arith.addi %global_splat, %r : tensor<128xi64>
    %base = tt.splat %output : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %ptr = tt.addptr %base, %offset : tensor<128x!tt.ptr<bf16>>, tensor<128xi64>
    tt.sr_store %ptr, %x, %index {site = "output"} : (tensor<128x!tt.ptr<bf16>>, tensor<128xf32>, tensor<128xi64>) -> ()
    tt.return
  }
}

// -----

module {
  tt.func public @sharded_without_grid(%seed: !tt.ptr<i64>, %base_seed: i64, %invocation: i64, %output: !tt.ptr<bf16>, %x: tensor<128xf32>) {
    %pid = tt.get_program_id x : i32
    %p = arith.extsi %pid : i32 to i64
    %c128 = arith.constant 128 : i64
    %c512 = arith.constant 512 : i64
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %r = arith.extsi %range : tensor<128xi32> to tensor<128xi64>
    %local = arith.muli %p, %c128 : i64
    %local_splat = tt.splat %local : i64 -> tensor<128xi64>
    %offset = arith.addi %local_splat, %r : tensor<128xi64>
    %global = arith.muli %p, %c512 : i64
    %shifted = arith.addi %global, %c128 : i64
    %global_splat = tt.splat %shifted : i64 -> tensor<128xi64>
    %index = arith.addi %global_splat, %r : tensor<128xi64>
    %base = tt.splat %output : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %ptr = tt.addptr %base, %offset : tensor<128x!tt.ptr<bf16>>, tensor<128xi64>
    // expected-error @+1 {{cannot prove injective matching pointer and logical indices}}
    tt.sr_store %ptr, %x, %index {site = "output"} : (tensor<128x!tt.ptr<bf16>>, tensor<128xf32>, tensor<128xi64>) -> ()
    tt.return
  }
}

// -----

module attributes {tt.sr_grid = 4 : i32} {
  tt.func private @sharded_in_helper(%seed: !tt.ptr<i64>, %base_seed: i64, %invocation: i64, %output: !tt.ptr<bf16>, %x: tensor<128xf32>) {
    %pid = tt.get_program_id x : i32
    %p = arith.extsi %pid : i32 to i64
    %c128 = arith.constant 128 : i64
    %c512 = arith.constant 512 : i64
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %r = arith.extsi %range : tensor<128xi32> to tensor<128xi64>
    %local = arith.muli %p, %c128 : i64
    %local_splat = tt.splat %local : i64 -> tensor<128xi64>
    %offset = arith.addi %local_splat, %r : tensor<128xi64>
    %global = arith.muli %p, %c512 : i64
    %shifted = arith.addi %global, %c128 : i64
    %global_splat = tt.splat %shifted : i64 -> tensor<128xi64>
    %index = arith.addi %global_splat, %r : tensor<128xi64>
    %base = tt.splat %output : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %ptr = tt.addptr %base, %offset : tensor<128x!tt.ptr<bf16>>, tensor<128xi64>
    // expected-error @+1 {{cannot prove injective matching pointer and logical indices}}
    tt.sr_store %ptr, %x, %index {site = "output"} : (tensor<128x!tt.ptr<bf16>>, tensor<128xf32>, tensor<128xi64>) -> ()
    tt.return
  }
}
