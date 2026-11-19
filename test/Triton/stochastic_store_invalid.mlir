// RUN: triton-opt %s --triton-materialize-stochastic-rounding='mode=enabled site=output' --split-input-file --verify-diagnostics
module {
  tt.func public @identity(%output: !tt.ptr<bf16>, %x: tensor<128xf32>) {
    %p = tt.get_program_id x : i32
    %p64 = arith.extui %p : i32 to i64
    %c128 = arith.constant 64 : i64
    %block = arith.muli %p64, %c128 : i64
    %splat = tt.splat %block : i64 -> tensor<128xi64>
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %r64 = arith.extui %range : tensor<128xi32> to tensor<128xi64>
    %index = arith.addi %splat, %r64 : tensor<128xi64>
    %base = tt.splat %output : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %ptr = tt.addptr %base, %index : tensor<128x!tt.ptr<bf16>>, tensor<128xi64>
    // expected-error @+1 {{cannot prove injective matching pointer and logical indices}}
    tt.sr_store %ptr, %x, %index {site = "output"} : (tensor<128x!tt.ptr<bf16>>, tensor<128xf32>, tensor<128xi64>) -> ()
    tt.return
  }
}

// -----

module {
  tt.func private @opaque(!tt.ptr<bf16>)
  tt.func public @identity(%output: !tt.ptr<bf16>, %x: tensor<128xf32>) {
    %p = tt.get_program_id x : i32
    %p64 = arith.extui %p : i32 to i64
    %c128 = arith.constant 128 : i64
    %block = arith.muli %p64, %c128 : i64
    %splat = tt.splat %block : i64 -> tensor<128xi64>
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %r64 = arith.extui %range : tensor<128xi32> to tensor<128xi64>
    %index = arith.addi %splat, %r64 : tensor<128xi64>
    %base = tt.splat %output : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %ptr = tt.addptr %base, %index : tensor<128x!tt.ptr<bf16>>, tensor<128xi64>
    tt.call @opaque(%output) : (!tt.ptr<bf16>) -> ()
    // expected-error @+1 {{cannot prove injective matching pointer and logical indices}}
    tt.sr_store %ptr, %x, %index {site = "output"} : (tensor<128x!tt.ptr<bf16>>, tensor<128xf32>, tensor<128xi64>) -> ()
    tt.return
  }
}
