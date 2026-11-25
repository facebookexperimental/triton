// RUN: triton-opt %s --triton-materialize-stochastic-rounding='mode=enabled site=output compute-capability=100 ptx-version=87 rounds=10' --convert-triton-to-tritongpu='target=cuda:100 num-warps=4' --tritongpu-assign-latencies --tritongpu-schedule-loops --tritongpu-pipeline=num-stages=3 --canonicalize | FileCheck %s --check-prefixes=CHECK,ENABLED --implicit-check-not=tt.sr_store
// RUN: triton-opt %s --triton-materialize-stochastic-rounding='mode=plain site=output' --convert-triton-to-tritongpu='target=cuda:100 num-warps=4' --tritongpu-assign-latencies --tritongpu-schedule-loops --tritongpu-pipeline=num-stages=3 --canonicalize | FileCheck %s --check-prefixes=CHECK,PLAIN --implicit-check-not=tt.sr_store

// tt.sr_store is materialized into an ordinary masked tt.store in make_ttir,
// before software pipelining. The loop is still pipelined, and the store keeps
// its mask, carried by the loop to the stage that executes the store.

// CHECK-LABEL: tt.func public @sr_store_loop
// CHECK-COUNT-2: ttg.async_copy_global_to_local {{.*}} -> <1024xf32
// CHECK: scf.for
// CHECK: ttg.async_wait
// CHECK: %[[VALUE:.*]] = ttg.local_load {{.*}} -> tensor<1024xf32
// ENABLED: %[[ROUNDED:.*]] = tt.fp_to_fp %[[VALUE]], rbits = {{.*}}rounding = rs
// PLAIN: %[[ROUNDED:.*]] = tt.fp_to_fp %[[VALUE]], rounding = rtne
// CHECK: tt.store %{{.*}}, %[[ROUNDED]], %arg{{[0-9]+}} : tensor<1024x!tt.ptr<bf16>
// CHECK: scf.yield
module {
  tt.func public @sr_store_loop(%seed: !tt.ptr<i64>, %base_seed: i64, %invocation: i64, %x: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %output: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %n: i64) {
    %c0 = arith.constant 0 : i64
    %c1024 = arith.constant 1024 : i64
    %c65536 = arith.constant 65536 : i64
    %pid = tt.get_program_id x : i32
    %pid64 = arith.extui %pid : i32 to i64
    %start = arith.muli %pid64, %c65536 : i64
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %r64 = arith.extui %range : tensor<1024xi32> to tensor<1024xi64>
    %bound = tt.splat %n : i64 -> tensor<1024xi64>
    %xs = tt.splat %x : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %outs = tt.splat %output : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    scf.for %iv = %c0 to %c65536 step %c1024 : i64 {
      %off = arith.addi %start, %iv : i64
      %offs = tt.splat %off : i64 -> tensor<1024xi64>
      %index = arith.addi %offs, %r64 : tensor<1024xi64>
      %mask = arith.cmpi slt, %index, %bound : tensor<1024xi64>
      %src = tt.addptr %xs, %index : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %value = tt.load %src, %mask : tensor<1024x!tt.ptr<f32>>
      %dst = tt.addptr %outs, %index : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi64>
      tt.sr_store %dst, %value, %index, %mask {site = "output"} : (tensor<1024x!tt.ptr<bf16>>, tensor<1024xf32>, tensor<1024xi64>, tensor<1024xi1>) -> ()
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}
