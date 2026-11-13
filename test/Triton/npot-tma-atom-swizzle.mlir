// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:100 -convert-triton-to-tritongpu="target=cuda:100 num-warps=8 threads-per-warp=32" -tritongpu-coalesce -tritongpu-accelerate-matmul 2>/dev/null | FileCheck %s --check-prefix=TC
// RUN: env -u TRITON_ALLOW_NPOT not triton-opt %s -split-input-file -triton-npot-pad=target=cuda:100 2>&1 | FileCheck %s --check-prefix=PAD

// A descriptor-fed N=144 dot stays exact at flag on, and B takes the pow2
// alloc shape's swizzle-128 layout (whole-atom box-64 TMA messages with
// overhang into the 256 alloc). Flag off rejects the NPOT descriptor before
// any layout is chosen, so the trigger is unreachable as upstream.
// TC-DAG: #[[$SW128:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
// TC-LABEL: @dot_desc_n144
// TC: ttg.local_alloc {{.*}} -> !ttg.memdesc<64x144xbf16, #[[$SW128]], {{.*}}>
// TC: ttng.tc_gen5_mma
// PAD: Number of elements must be power-of-two
tt.func @dot_desc_n144(%a_ptr: !tt.ptr<bf16>, %b_ptr: !tt.ptr<bf16>, %c_ptr: !tt.ptr<f32>) {
  %m = arith.constant 4096 : i32
  %n = arith.constant 144 : i32
  %k = arith.constant 4096 : i32
  %s = arith.constant 4096 : i64
  %c1 = arith.constant 1 : i64
  %c0 = arith.constant 0 : i32
  %db = tt.make_tensor_descriptor %b_ptr, [%k, %n], [%s, %c1] : !tt.ptr<bf16>, !tt.tensordesc<64x144xbf16>
  %b = tt.descriptor_load %db[%c0, %c0] : !tt.tensordesc<64x144xbf16> -> tensor<64x144xbf16>
  %sa = tt.splat %a_ptr : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x64x!tt.ptr<bf16>>
  %c = arith.constant dense<0.000000e+00> : tensor<128x144xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x144xbf16> -> tensor<128x144xf32>
  %sc = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<128x144x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x144x!tt.ptr<f32>>
  tt.return
}

// -----

// N=48 is narrower than one 128B swizzle span (64 f16): the trigger must
// not fire (the 128B layout cannot be built), B keeps swizzle 32 and the
// dot still accelerates.
// TC-DAG: #[[$SW32:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
// TC-LABEL: @dot_desc_n48
// TC: ttg.local_alloc {{.*}} -> !ttg.memdesc<64x48xbf16, #[[$SW32]], {{.*}}>
// TC: ttng.tc_gen5_mma
tt.func @dot_desc_n48(%a_ptr: !tt.ptr<bf16>, %b_ptr: !tt.ptr<bf16>, %c_ptr: !tt.ptr<f32>) {
  %m = arith.constant 4096 : i32
  %n = arith.constant 48 : i32
  %k = arith.constant 4096 : i32
  %s = arith.constant 4096 : i64
  %c1 = arith.constant 1 : i64
  %c0 = arith.constant 0 : i32
  %db = tt.make_tensor_descriptor %b_ptr, [%k, %n], [%s, %c1] : !tt.ptr<bf16>, !tt.tensordesc<64x48xbf16>
  %b = tt.descriptor_load %db[%c0, %c0] : !tt.tensordesc<64x48xbf16> -> tensor<64x48xbf16>
  %sa = tt.splat %a_ptr : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
  %a = tt.load %sa : tensor<128x64x!tt.ptr<bf16>>
  %c = arith.constant dense<0.000000e+00> : tensor<128x48xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x48xbf16> -> tensor<128x48xf32>
  %sc = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<128x48x!tt.ptr<f32>>
  tt.store %sc, %d : tensor<128x48x!tt.ptr<f32>>
  tt.return
}
