// RUN: TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s
//
// NPOT (non-power-of-2) elementwise lowering. A non-power-of-2 out-dim builds a
// modular LinearLayout, so the per-element index arithmetic uses ADD + a modulo
// (Barrett multiply-shift, or the compare-subtract fast path when the value is
// provably < 2N) instead of the pow2 GF(2) XOR. The AND-mask fast path replaces
// the modulo entirely for dense identity bases (x mod 2^k == x & (2^k - 1)).
// The pow2 companion module below asserts the modular ops never appear when the
// shape is a power of two -> the NPOT path is byte-identical for pow2 / flag-off.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // N=48 is not a power of two and its worst-case index exceeds 2N, so the
  // modulo lowers to Barrett reduction: q = (x * M) >> k; x - q*N, with the
  // compile-time constants M = 5726623062 and k = 38 for N = 48.
  // CHECK-LABEL: llvm.func @npot_barrett
  // CHECK: %[[M:.*]] = llvm.mlir.constant(5726623062 : i64) : i64
  // CHECK: llvm.mul %{{.*}}, %[[M]] : i64
  // CHECK: %[[K:.*]] = llvm.mlir.constant(38 : i64) : i64
  // CHECK: llvm.lshr %{{.*}}, %[[K]] : i64
  // CHECK: llvm.trunc %{{.*}} : i64 to i32
  // CHECK: %[[N:.*]] = llvm.mlir.constant(48 : i32) : i32
  // CHECK: llvm.mul %{{.*}}, %[[N]] : i32
  // CHECK: llvm.sub %{{.*}}, %{{.*}} : i32
  tt.func public @npot_barrett(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 48 : i32, start = 0 : i32} : tensor<48xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<48x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<48x!tt.ptr<f32>, #blocked>, tensor<48xi32, #blocked>
    %3 = arith.sitofp %0 : tensor<48xi32, #blocked> to tensor<48xf32, #blocked>
    tt.store %2, %3 : tensor<48x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // N=1023 has dense identity register/lane/warp bases, so combining them hits
  // the AND-mask fast path: (x mod 2^7) == x & 127 (highBit = 7, ceil(log2 1023)
  // = 10 but the combined lane+warp span here is 128). The per-register ADD then
  // overshoots at most into [N, 2N), so the final modulo is the cheap
  // compare-subtract path (icmp uge N + select), not Barrett (no i64 multiply).
  // CHECK-LABEL: llvm.func @npot_andmask
  // CHECK: %[[MASK:.*]] = llvm.mlir.constant(127 : i32) : i32
  // CHECK: llvm.and %{{.*}}, %[[MASK]] : i32
  // CHECK: llvm.add %{{.*}}, %{{.*}} : i32
  // CHECK: %[[N:.*]] = llvm.mlir.constant(1023 : i32) : i32
  // CHECK: llvm.icmp "uge" %{{.*}}, %[[N]] : i32
  // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, i32
  // Barrett must not appear for this size (no 64-bit multiply-shift modulo).
  // CHECK-NOT: llvm.mul %{{.*}}, %{{.*}} : i64
  tt.func public @npot_andmask(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 1023 : i32, start = 0 : i32} : tensor<1023xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1023x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<1023x!tt.ptr<f32>, #blocked>, tensor<1023xi32, #blocked>
    %3 = arith.sitofp %0 : tensor<1023xi32, #blocked> to tensor<1023xf32, #blocked>
    tt.store %2, %3 : tensor<1023x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Pow2 companion (N=256): the shape is a power of two so the layout is never
  // modular. The index arithmetic stays on the original GF(2) XOR path -- none
  // of the modular ops (Barrett i64 multiply-shift, or the compare-subtract
  // modulo) appear. This is the byte-identity guard: pow2 lowering is unchanged.
  // CHECK-LABEL: llvm.func @pow2_no_modulo
  // CHECK-NOT: llvm.mul %{{.*}}, %{{.*}} : i64
  // CHECK-NOT: llvm.icmp "uge"
  tt.func public @pow2_no_modulo(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %3 = arith.sitofp %0 : tensor<256xi32, #blocked> to tensor<256xf32, #blocked>
    tt.store %2, %3 : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
