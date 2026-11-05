// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' -tritongpu-coalesce -tritongpu-remove-layout-conversions -allocate-shared-memory-nv=compute-capability=90 -convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s --check-prefix=W4
// RUN: env TRITON_ALLOW_NPOT=1 triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:90 num-warps=1' -tritongpu-coalesce -tritongpu-remove-layout-conversions -allocate-shared-memory-nv=compute-capability=90 -convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s --check-prefix=W1

// A Product load takes its power-of-two control's vector: the [96, 64] tile at
// four warps and 384 at one warp both load v4, as [128, 64] and 512 do.
// W4-LABEL: @copy_96x64
// W4-COUNT-12: llvm.inline_asm {{.*}}ld.global.v4.b32
// W4-NOT: llvm.load {{.*}} !llvm.ptr<1>
// W4: llvm.return
tt.func @copy_96x64(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %r = tt.make_range {end = 6144 : i32, start = 0 : i32} : tensor<6144xi32>
  %i = tt.reshape %r : tensor<6144xi32> -> tensor<96x64xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<96x64x!tt.ptr<f32>>
  %p = tt.addptr %sp, %i : tensor<96x64x!tt.ptr<f32>>, tensor<96x64xi32>
  %x = tt.load %p : tensor<96x64x!tt.ptr<f32>>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<96x64x!tt.ptr<f32>>
  %q = tt.addptr %dp, %i : tensor<96x64x!tt.ptr<f32>>, tensor<96x64xi32>
  tt.store %q, %x : tensor<96x64x!tt.ptr<f32>>
  tt.return
}

// -----

// W4-LABEL: @copy_128x64
// W4-COUNT-16: llvm.inline_asm {{.*}}ld.global.v4.b32
// W4-NOT: llvm.load {{.*}} !llvm.ptr<1>
// W4: llvm.return
tt.func @copy_128x64(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %r = tt.make_range {end = 8192 : i32, start = 0 : i32} : tensor<8192xi32>
  %i = tt.reshape %r : tensor<8192xi32> -> tensor<128x64xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>>
  %p = tt.addptr %sp, %i : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
  %x = tt.load %p : tensor<128x64x!tt.ptr<f32>>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>>
  %q = tt.addptr %dp, %i : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xi32>
  tt.store %q, %x : tensor<128x64x!tt.ptr<f32>>
  tt.return
}

// -----

// W1-LABEL: @copy_384
// W1-COUNT-3: llvm.inline_asm {{.*}}ld.global.v4.b32
// W1-NOT: llvm.load {{.*}} !llvm.ptr<1>
// W1: llvm.return
tt.func @copy_384(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  %x = tt.load %p : tensor<384x!tt.ptr<f32>>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<384x!tt.ptr<f32>>
  %q = tt.addptr %dp, %r : tensor<384x!tt.ptr<f32>>, tensor<384xi32>
  tt.store %q, %x : tensor<384x!tt.ptr<f32>>
  tt.return
}

// -----

// W1-LABEL: @copy_512
// W1-COUNT-4: llvm.inline_asm {{.*}}ld.global.v4.b32
// W1-NOT: llvm.load {{.*}} !llvm.ptr<1>
// W1: llvm.return
tt.func @copy_512(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %r = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
  %sp = tt.splat %src : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
  %p = tt.addptr %sp, %r : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
  %x = tt.load %p : tensor<512x!tt.ptr<f32>>
  %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
  %q = tt.addptr %dp, %r : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
  tt.store %q, %x : tensor<512x!tt.ptr<f32>>
  tt.return
}
