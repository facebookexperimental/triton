// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad='num-warps=4' -verify-diagnostics | FileCheck %s

#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#unswizzled = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#swizzled128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

// An explicit shared allocation keeps its exact dense shape where a
// shared encoding for it is legal (unswizzled 104 here).
// CHECK-LABEL: @local_alloc_104_exact
tt.func @local_alloc_104_exact(%pa: !tt.ptr<f16>, %po: !tt.ptr<f16>) {
  // CHECK: ttg.local_alloc %{{.*}} : (tensor<128x104xf16>) -> !ttg.memdesc<128x104xf16, #{{.*}}, #{{.*}}, mutable>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x104x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<128x104x!tt.ptr<f16>>
  %m = ttg.local_alloc %a : (tensor<128x104xf16>) -> !ttg.memdesc<128x104xf16, #unswizzled, #smem, mutable>
  %v = ttg.local_load %m : !ttg.memdesc<128x104xf16, #unswizzled, #smem, mutable> -> tensor<128x104xf16>
  %so = tt.splat %po : !tt.ptr<f16> -> tensor<128x104x!tt.ptr<f16>>
  tt.store %so, %v : tensor<128x104x!tt.ptr<f16>>
  tt.return
}

// -----

#smem2 = #ttg.shared_memory
#swizzled128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

// Where the encoding forces whole atoms (128B swizzle on 104), the
// allocation takes the pow2 cover.
// CHECK-LABEL: @local_alloc_104_cover
tt.func @local_alloc_104_cover(%pa: !tt.ptr<f16>, %po: !tt.ptr<f16>) {
  // CHECK: ttg.local_alloc %{{.*}} : (tensor<128x128xf16>) -> !ttg.memdesc<128x128xf16, #{{.*}}, #{{.*}}, mutable>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x104x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<128x104x!tt.ptr<f16>>
  // expected-remark @+1 {{NPOT shared allocation rows do not tile whole swizzle atoms}}
  %m = ttg.local_alloc %a : (tensor<128x104xf16>) -> !ttg.memdesc<128x104xf16, #swizzled128, #smem2, mutable>
  %v = ttg.local_load %m : !ttg.memdesc<128x104xf16, #swizzled128, #smem2, mutable> -> tensor<128x104xf16>
  %so = tt.splat %po : !tt.ptr<f16> -> tensor<128x104x!tt.ptr<f16>>
  tt.store %so, %v : tensor<128x104x!tt.ptr<f16>>
  tt.return
}

// -----

#smem3 = #ttg.shared_memory
#unswizzled3 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>

// Unswizzled rows must still span a multiple of 16B (100 x 2B = 200B does
// not), so this allocation pads to the pow2 cover.
// CHECK-LABEL: @local_alloc_100_cover
tt.func @local_alloc_100_cover(%pa: !tt.ptr<f16>, %po: !tt.ptr<f16>) {
  // CHECK: ttg.local_alloc %{{.*}} : (tensor<128x128xf16>) -> !ttg.memdesc<128x128xf16, #{{.*}}, #{{.*}}, mutable>
  %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x100x!tt.ptr<f16>>
  %a = tt.load %sa : tensor<128x100x!tt.ptr<f16>>
  // expected-remark @+1 {{NPOT shared allocation rows do not tile whole swizzle atoms}}
  %m = ttg.local_alloc %a : (tensor<128x100xf16>) -> !ttg.memdesc<128x100xf16, #unswizzled3, #smem3, mutable>
  %v = ttg.local_load %m : !ttg.memdesc<128x100xf16, #unswizzled3, #smem3, mutable> -> tensor<128x100xf16>
  %so = tt.splat %po : !tt.ptr<f16> -> tensor<128x100x!tt.ptr<f16>>
  tt.store %so, %v : tensor<128x100x!tt.ptr<f16>>
  tt.return
}
