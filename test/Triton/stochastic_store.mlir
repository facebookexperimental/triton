// RUN: triton-opt %s --split-input-file --triton-materialize-stochastic-rounding='mode=plain site=output' | FileCheck %s
// RUN: triton-opt %s --split-input-file | triton-opt --split-input-file | FileCheck %s --check-prefix=ROUNDTRIP

// CHECK-LABEL: @identity
// CHECK-NOT: tt.sr_store
// CHECK: tt.fp_to_fp {{.*}} rounding = rtne
// CHECK: tt.store
// ROUNDTRIP: tt.sr_store
module {
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
    tt.sr_store %ptr, %x, %index {site = "output"} : (tensor<128x!tt.ptr<bf16>>, tensor<128xf32>, tensor<128xi64>) -> ()
    tt.return
  }
}

// -----

// CHECK-LABEL: @masked_cache
// CHECK: tt.fp_to_fp {{.*}} rounding = rtne
// CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}} {cachePolicy = #tt.cache_policy<cache_modifier = cs, eviction_policy = evict_last>}
// ROUNDTRIP-LABEL: @masked_cache
// ROUNDTRIP: tt.sr_store
module {
  tt.func public @masked_cache(%output: !tt.ptr<bf16>, %x: f32, %index: i64, %mask: i1) {
    tt.sr_store %output, %x, %index, %mask {site = "output", cache = 5 : i32, evict = 3 : i32} : (!tt.ptr<bf16>, f32, i64, i1) -> ()
    tt.return
  }
}
