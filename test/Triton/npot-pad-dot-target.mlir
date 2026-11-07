// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:75 -verify-diagnostics | FileCheck %s
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=hip:gfx950 -tritonamdgpu-schedule-loops -o /dev/null 2>&1 | FileCheck %s --check-prefix=HIP
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad -o /dev/null 2>&1 | FileCheck %s --check-prefix=NONE
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:90 2>/dev/null | FileCheck %s --check-prefix=ATTRS
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -triton-npot-pad=target=cuda:90 -convert-triton-to-tritongpu=target=cuda:90 2>/dev/null | FileCheck %s --check-prefix=TTG

// The target's dialect names why the dot pads. Its NPOT dim is on operand B.
// The AMD dialect loads with its backend, so the HIP RUN adds an AMD pass.
// CHECK-LABEL: @dot_n96
// CHECK: tt.dot {{.*}} -> tensor<128x128xf32>
// HIP: remark: NPOT dot is not yet supported with 2 warps of 32 on dim 1, whose power-of-two part is 32
// NONE: remark: NPOT dot is not yet supported without an MMA target
// ConvertTritonToTritonGPU, not the pad pass, sets the module attrs.
// ATTRS: module {
// ATTRS-NEXT: tt.func @dot_n96
// TTG: module attributes {{.*}}"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"
// TTG-NEXT: tt.func @dot_n96
tt.func @dot_n96(%pb: !tt.ptr<f16>) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x96x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x96x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x96xf32>
  // expected-remark @+1 {{NPOT dot is not yet supported on cuda:75}}
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x96xf16> -> tensor<128x96xf32>
  tt.return
}

// -----

// CHECK-LABEL: @dot_pow2
// CHECK: tt.dot {{.*}} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
tt.func @dot_pow2(%pb: !tt.ptr<f16>) {
  %a = arith.constant dense<1.0> : tensor<128x64xf16>
  %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
  %b = tt.load %sb : tensor<64x128x!tt.ptr<f16>>
  %c = arith.constant dense<0.0> : tensor<128x128xf32>
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
  tt.return
}

// -----

// The pass reads only its target option; the module's ttg.target belongs to
// ConvertTritonToTritonGPU.
// CHECK-LABEL: @dot_n96_module_target
// CHECK: tt.dot {{.*}} -> tensor<128x128xf32>
// NONE: remark: NPOT dot is not yet supported without an MMA target
module attributes {ttg.target = "cuda:90"} {
  tt.func @dot_n96_module_target(%pb: !tt.ptr<f16>) {
    %a = arith.constant dense<1.0> : tensor<128x64xf16>
    %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x96x!tt.ptr<f16>>
    %b = tt.load %sb : tensor<64x96x!tt.ptr<f16>>
    %c = arith.constant dense<0.0> : tensor<128x96xf32>
    // expected-remark @+1 {{NPOT dot is not yet supported on cuda:75}}
    %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x96xf16> -> tensor<128x96xf32>
    tt.return
  }
}

// -----

// The target names why a descriptor pads. On cuda its 3 TMA messages against 2
// pad it.
// CHECK-LABEL: @descriptor_64x96
// CHECK: tt.descriptor_load {{.*}} -> tensor<64x128xf16>
// HIP: remark: NPOT tensor descriptor is not yet supported on hip:gfx950
// NONE: remark: NPOT tensor descriptor is not yet supported without a target
tt.func @descriptor_64x96(%a: !tt.ptr<f16>, %c: !tt.ptr<f16>, %m: i32, %n: i32, %s: i64) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %da = tt.make_tensor_descriptor %a, [%m, %n], [%s, %c1] : !tt.ptr<f16>, !tt.tensordesc<64x96xf16>
  %x = tt.descriptor_load %da[%c0, %c0] : !tt.tensordesc<64x96xf16> -> tensor<64x96xf16>
  %sc = tt.splat %c : !tt.ptr<f16> -> tensor<64x96x!tt.ptr<f16>>
  tt.store %sc, %x : tensor<64x96x!tt.ptr<f16>>
  tt.return
}
