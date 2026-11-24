// RUN: split-file %s %t
// RUN: triton-opt %t/cache.mlir --triton-materialize-stochastic-rounding='mode=plain site=output' | FileCheck %s --check-prefix=CACHE
// RUN: triton-opt %t/cache.mlir | triton-opt | FileCheck %s --check-prefix=CACHE-ROUNDTRIP
// RUN: triton-opt %t/plain.mlir --triton-materialize-stochastic-rounding='mode=plain site=output' | FileCheck %s --check-prefix=PLAIN --implicit-check-not=tt.sr_store --implicit-check-not='rounding = rs' --implicit-check-not=tt.load --implicit-check-not=tt.mulhiui --implicit-check-not=sr_ieee --implicit-check-not=tt.fp_to_fp --implicit-check-not=tt.store --implicit-check-not=arith.truncf
// RUN: triton-opt %t/plain.mlir | triton-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: triton-opt %t/enabled.mlir --triton-materialize-stochastic-rounding='mode=enabled site=output compute-capability=100 ptx-version=87 rounds=7' | FileCheck %s --check-prefixes=ENABLED,R7 --implicit-check-not=tt.sr_store --implicit-check-not='rounding = rtne' --implicit-check-not='rounding = rtz' --implicit-check-not=arith.truncf --implicit-check-not=tt.load --implicit-check-not=tt.store --implicit-check-not=tt.fp_to_fp --implicit-check-not=tt.mulhiui
// RUN: triton-opt %t/enabled.mlir --triton-materialize-stochastic-rounding='mode=enabled site=output compute-capability=100 ptx-version=87 rounds=10' | FileCheck %s --check-prefixes=ENABLED,R10 --implicit-check-not=tt.sr_store --implicit-check-not='rounding = rtne' --implicit-check-not='rounding = rtz' --implicit-check-not=arith.truncf --implicit-check-not=tt.load --implicit-check-not=tt.store --implicit-check-not=tt.fp_to_fp --implicit-check-not=tt.mulhiui
// RUN: triton-opt %t/enabled.mlir --triton-materialize-stochastic-rounding='mode=enabled site=output compute-capability=103 ptx-version=87 rounds=7' | FileCheck %s --check-prefixes=ENABLED,R7 --implicit-check-not=tt.sr_store --implicit-check-not='rounding = rtne' --implicit-check-not='rounding = rtz' --implicit-check-not=arith.truncf --implicit-check-not=tt.load --implicit-check-not=tt.store --implicit-check-not=tt.fp_to_fp --implicit-check-not=tt.mulhiui
// RUN: triton-opt %t/enabled.mlir --triton-materialize-stochastic-rounding='mode=enabled site=output compute-capability=103 ptx-version=87 rounds=10' | FileCheck %s --check-prefixes=ENABLED,R10 --implicit-check-not=tt.sr_store --implicit-check-not='rounding = rtne' --implicit-check-not='rounding = rtz' --implicit-check-not=arith.truncf --implicit-check-not=tt.load --implicit-check-not=tt.store --implicit-check-not=tt.fp_to_fp --implicit-check-not=tt.mulhiui

// Implicit negatives cover the entire output, including gaps between checks.
// In enabled mode, explicit matches permit exactly one seed load, conversion
// and store, and exactly two unsigned high multiplies per Philox round.

// PLAIN-LABEL: tt.func public @identity(
// PLAIN-SAME: %[[OUTPUT:[a-zA-Z0-9_]+]]: !tt.ptr<bf16>, %[[X:[a-zA-Z0-9_]+]]: tensor<128xf32>)
// PLAIN-NOT: tt.sr_store
// PLAIN: %[[PTR:[a-zA-Z0-9_]+]] = tt.addptr
// PLAIN: %[[BF16:[a-zA-Z0-9_]+]] = tt.fp_to_fp %[[X]], rounding = rtne : tensor<128xf32> -> tensor<128xbf16>
// PLAIN-NEXT: tt.store %[[PTR]], %[[BF16]] : tensor<128x!tt.ptr<bf16>>
// PLAIN-NEXT: tt.return
// PLAIN-NEXT: }
// PLAIN-NEXT: }
// ROUNDTRIP: tt.sr_store

// ENABLED-LABEL: tt.func public @identity(
// ENABLED-SAME: %[[SEED:[a-zA-Z0-9_]+]]: !tt.ptr<i64>, %[[BASE:[a-zA-Z0-9_]+]]: i64, %[[INVOCATION:[a-zA-Z0-9_]+]]: i64, %[[OUTPUT:[a-zA-Z0-9_]+]]: !tt.ptr<bf16>, %[[X:[a-zA-Z0-9_]+]]: tensor<128xf32>)
// ENABLED-NEXT: %[[DYNAMIC:[a-zA-Z0-9_]+]] = tt.load %[[SEED]] : !tt.ptr<i64>
// ENABLED-NEXT: %[[SEED_DOMAIN:[a-zA-Z0-9_]+]] = arith.constant 8390885004551279361 : i64
// ENABLED-NEXT: {{.*}} = arith.xori %[[BASE]], %[[SEED_DOMAIN]] : i64
// ENABLED: {{.*}} = arith.xori {{.*}}, %[[DYNAMIC]] : i64
// ENABLED: %[[SEED_MULTIPLIER:[-a-zA-Z0-9_]+]] = arith.constant -4265267296055464877 : i64
// ENABLED-NEXT: %[[SEED_PRODUCT:[a-zA-Z0-9_]+]] = arith.muli {{.*}}, %[[SEED_MULTIPLIER]] : i64
// ENABLED-NEXT: %[[SEED_SHIFT:[a-zA-Z0-9_]+]] = arith.constant 33 : i64
// ENABLED-NEXT: %[[SEED_SHIFTED:[a-zA-Z0-9_]+]] = arith.shrui %[[SEED_PRODUCT]], %[[SEED_SHIFT]] : i64
// ENABLED-NEXT: %[[SEED_HASH:[a-zA-Z0-9_]+]] = arith.xori %[[SEED_PRODUCT]], %[[SEED_SHIFTED]] : i64
// ENABLED-NEXT: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// ENABLED: %[[INVOCATION_DOMAIN:[a-zA-Z0-9_]+]] = arith.constant 8390885004551279362 : i64
// ENABLED-NEXT: {{.*}} = arith.xori %[[INVOCATION]], %[[INVOCATION_DOMAIN]] : i64
// ENABLED: {{.*}} = arith.xori {{.*}}, %[[ZERO]] : i64
// ENABLED: %[[INVOCATION_MULTIPLIER:[-a-zA-Z0-9_]+]] = arith.constant -4265267296055464877 : i64
// ENABLED-NEXT: %[[INVOCATION_PRODUCT:[a-zA-Z0-9_]+]] = arith.muli {{.*}}, %[[INVOCATION_MULTIPLIER]] : i64
// ENABLED-NEXT: %[[INVOCATION_SHIFT:[a-zA-Z0-9_]+]] = arith.constant 33 : i64
// ENABLED-NEXT: %[[INVOCATION_SHIFTED:[a-zA-Z0-9_]+]] = arith.shrui %[[INVOCATION_PRODUCT]], %[[INVOCATION_SHIFT]] : i64
// ENABLED-NEXT: %[[INVOCATION_HASH:[a-zA-Z0-9_]+]] = arith.xori %[[INVOCATION_PRODUCT]], %[[INVOCATION_SHIFTED]] : i64
// ENABLED: %[[PID:[a-zA-Z0-9_]+]] = tt.get_program_id x : i32
// ENABLED-NEXT: %[[PID64:[a-zA-Z0-9_]+]] = arith.extui %[[PID]] : i32 to i64
// ENABLED-NEXT: %[[BLOCK_SIZE:[a-zA-Z0-9_]+]] = arith.constant 128 : i64
// ENABLED-NEXT: %[[BLOCK:[a-zA-Z0-9_]+]] = arith.muli %[[PID64]], %[[BLOCK_SIZE]] : i64
// ENABLED-NEXT: %[[BLOCKS:[a-zA-Z0-9_]+]] = tt.splat %[[BLOCK]] : i64 -> tensor<128xi64>
// ENABLED-NEXT: %[[RANGE:[a-zA-Z0-9_]+]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// ENABLED-NEXT: %[[RANGE64:[a-zA-Z0-9_]+]] = arith.extui %[[RANGE]] : tensor<128xi32> to tensor<128xi64>
// ENABLED-NEXT: %[[INDEX:[a-zA-Z0-9_]+]] = arith.addi %[[BLOCKS]], %[[RANGE64]] : tensor<128xi64>
// ENABLED-NEXT: %[[POINTER_BASE:[a-zA-Z0-9_]+]] = tt.splat %[[OUTPUT]] : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
// ENABLED-NEXT: %[[PTR:[a-zA-Z0-9_]+]] = tt.addptr %[[POINTER_BASE]], %[[INDEX]] : tensor<128x!tt.ptr<bf16>>, tensor<128xi64>
// ENABLED-NEXT: %[[LIMIT:[a-zA-Z0-9_]+]] = arith.constant dense<120> : tensor<128xi32>
// ENABLED-NEXT: %[[MASK:[a-zA-Z0-9_]+]] = arith.cmpi ult, %[[RANGE]], %[[LIMIT]] : tensor<128xi32>

// The logical index supplies counter words 0/1; the hashed invocation supplies
// words 2/3. The independently hashed base/device seed supplies key words 0/1.
// ENABLED-NEXT: %[[C0:[a-zA-Z0-9_]+]] = arith.trunci %[[INDEX]] : tensor<128xi64> to tensor<128xi32>
// ENABLED-NEXT: %[[SHIFT_INDEX:[a-zA-Z0-9_]+]] = arith.constant dense<32> : tensor<128xi64>
// ENABLED-NEXT: %[[INDEX_HIGH:[a-zA-Z0-9_]+]] = arith.shrui %[[INDEX]], %[[SHIFT_INDEX]] : tensor<128xi64>
// ENABLED-NEXT: %[[C1:[a-zA-Z0-9_]+]] = arith.trunci %[[INDEX_HIGH]] : tensor<128xi64> to tensor<128xi32>
// ENABLED-NEXT: %[[INVOCATION_LOW:[a-zA-Z0-9_]+]] = arith.trunci %[[INVOCATION_HASH]] : i64 to i32
// ENABLED-NEXT: %[[C2:[a-zA-Z0-9_]+]] = tt.splat %[[INVOCATION_LOW]] : i32 -> tensor<128xi32>
// ENABLED-NEXT: %[[SHIFT_INVOCATION:[a-zA-Z0-9_]+]] = arith.constant 32 : i64
// ENABLED-NEXT: %[[INVOCATION_HIGH64:[a-zA-Z0-9_]+]] = arith.shrui %[[INVOCATION_HASH]], %[[SHIFT_INVOCATION]] : i64
// ENABLED-NEXT: %[[INVOCATION_HIGH:[a-zA-Z0-9_]+]] = arith.trunci %[[INVOCATION_HIGH64]] : i64 to i32
// ENABLED-NEXT: %[[C3:[a-zA-Z0-9_]+]] = tt.splat %[[INVOCATION_HIGH]] : i32 -> tensor<128xi32>
// ENABLED-NEXT: %[[K0:[a-zA-Z0-9_]+]] = arith.trunci %[[SEED_HASH]] : i64 to i32
// ENABLED-NEXT: %[[SHIFT_SEED:[a-zA-Z0-9_]+]] = arith.constant 32 : i64
// ENABLED-NEXT: %[[SEED_HIGH64:[a-zA-Z0-9_]+]] = arith.shrui %[[SEED_HASH]], %[[SHIFT_SEED]] : i64
// ENABLED-NEXT: %[[K1:[a-zA-Z0-9_]+]] = arith.trunci %[[SEED_HIGH64]] : i64 to i32
// ENABLED-NEXT: %[[M0:[a-zA-Z0-9_]+]] = arith.constant dense<-766435501> : tensor<128xi32>
// ENABLED-NEXT: %[[M1:[a-zA-Z0-9_]+]] = arith.constant dense<-845247145> : tensor<128xi32>

// Check the first round's counter/key wiring and the Weyl key increments.
// ENABLED-NEXT: %[[LO0:[a-zA-Z0-9_]+]] = arith.muli %[[C0]], %[[M0]] : tensor<128xi32>
// ENABLED-NEXT: %[[HI0:[a-zA-Z0-9_]+]] = tt.mulhiui %[[C0]], %[[M0]] : tensor<128xi32>
// ENABLED-NEXT: %[[LO1:[a-zA-Z0-9_]+]] = arith.muli %[[C2]], %[[M1]] : tensor<128xi32>
// ENABLED-NEXT: %[[HI1:[a-zA-Z0-9_]+]] = tt.mulhiui %[[C2]], %[[M1]] : tensor<128xi32>
// ENABLED-NEXT: %[[FIRST_XOR0:[a-zA-Z0-9_]+]] = arith.xori %[[HI1]], %[[C1]] : tensor<128xi32>
// ENABLED-NEXT: %[[FIRST_K0:[a-zA-Z0-9_]+]] = tt.splat %[[K0]] : i32 -> tensor<128xi32>
// ENABLED-NEXT: {{.*}} = arith.xori %[[FIRST_XOR0]], %[[FIRST_K0]] : tensor<128xi32>
// ENABLED-NEXT: %[[FIRST_XOR2:[a-zA-Z0-9_]+]] = arith.xori %[[HI0]], %[[C3]] : tensor<128xi32>
// ENABLED-NEXT: %[[FIRST_K1:[a-zA-Z0-9_]+]] = tt.splat %[[K1]] : i32 -> tensor<128xi32>
// ENABLED-NEXT: {{.*}} = arith.xori %[[FIRST_XOR2]], %[[FIRST_K1]] : tensor<128xi32>
// ENABLED-NEXT: %[[W0:[-a-zA-Z0-9_]+]] = arith.constant -1640531527 : i32
// ENABLED-NEXT: {{.*}} = arith.addi %[[K0]], %[[W0]] : i32
// ENABLED-NEXT: %[[W1:[-a-zA-Z0-9_]+]] = arith.constant -1150833019 : i32
// ENABLED-NEXT: {{.*}} = arith.addi %[[K1]], %[[W1]] : i32

// Five/eight middle rounds separate the first and final round.
// R7-COUNT-10: = tt.mulhiui
// R10-COUNT-16: = tt.mulhiui
// ENABLED: %[[LAST_LO0:[a-zA-Z0-9_]+]] = arith.muli %[[LAST_C0:[a-zA-Z0-9_]+]], %[[M0]] : tensor<128xi32>
// ENABLED-NEXT: %[[LAST_HI0:[a-zA-Z0-9_]+]] = tt.mulhiui %[[LAST_C0]], %[[M0]] : tensor<128xi32>
// ENABLED-NEXT: %[[LAST_LO1:[a-zA-Z0-9_]+]] = arith.muli %[[LAST_C2:[a-zA-Z0-9_]+]], %[[M1]] : tensor<128xi32>
// ENABLED-NEXT: %[[LAST_HI1:[a-zA-Z0-9_]+]] = tt.mulhiui %[[LAST_C2]], %[[M1]] : tensor<128xi32>
// ENABLED-NEXT: %[[LAST_XOR0:[a-zA-Z0-9_]+]] = arith.xori %[[LAST_HI1]], {{.*}} : tensor<128xi32>
// ENABLED-NEXT: %[[LAST_K0:[a-zA-Z0-9_]+]] = tt.splat {{.*}} : i32 -> tensor<128xi32>
// ENABLED-NEXT: %[[R0:[a-zA-Z0-9_]+]] = arith.xori %[[LAST_XOR0]], %[[LAST_K0]] : tensor<128xi32>
// ENABLED-NEXT: %[[LAST_XOR2:[a-zA-Z0-9_]+]] = arith.xori %[[LAST_HI0]], {{.*}} : tensor<128xi32>
// ENABLED-NEXT: %[[LAST_K1:[a-zA-Z0-9_]+]] = tt.splat {{.*}} : i32 -> tensor<128xi32>
// ENABLED-NEXT: {{.*}} = arith.xori %[[LAST_XOR2]], %[[LAST_K1]] : tensor<128xi32>
// ENABLED-NEXT: %[[BF16:[a-zA-Z0-9_]+]] = tt.fp_to_fp %[[X]], rbits = %[[R0]] : tensor<128xi32>, rounding = rs {sr_ieee} : tensor<128xf32> -> tensor<128xbf16>
// ENABLED-NEXT: tt.store %[[PTR]], %[[BF16]], %[[MASK]] : tensor<128x!tt.ptr<bf16>>
// ENABLED-NEXT: tt.return
// ENABLED-NEXT: }
// ENABLED-NEXT: }

//--- plain.mlir
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

//--- enabled.mlir
module {
  tt.func public @identity(%seed: !tt.ptr<i64>, %base_seed: i64, %invocation: i64, %output: !tt.ptr<bf16>, %x: tensor<128xf32>) {
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
    %limit = arith.constant dense<120> : tensor<128xi32>
    %mask = arith.cmpi ult, %range, %limit : tensor<128xi32>
    tt.sr_store %ptr, %x, %index, %mask {site = "output"} : (tensor<128x!tt.ptr<bf16>>, tensor<128xf32>, tensor<128xi64>, tensor<128xi1>) -> ()
    tt.return
  }
}

//--- cache.mlir
// CACHE-LABEL: @masked_cache
// CACHE: tt.fp_to_fp {{.*}} rounding = rtne
// CACHE: tt.store %{{.*}}, %{{.*}}, %{{.*}} {cachePolicy = #tt.cache_policy<cache_modifier = cs, eviction_policy = evict_last>}
// CACHE-ROUNDTRIP-LABEL: @masked_cache
// CACHE-ROUNDTRIP: tt.sr_store
module {
  tt.func public @masked_cache(%output: !tt.ptr<bf16>, %x: f32, %index: i64, %mask: i1) {
    tt.sr_store %output, %x, %index, %mask {site = "output", cache = 5 : i32, evict = 3 : i32} : (!tt.ptr<bf16>, f32, i64, i1) -> ()
    tt.return
  }
}
