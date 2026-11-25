// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 --verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 1024 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mask_limits_direct_to_lds_width(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %offsets: tensor<64x4xi32, #blocked> {tt.contiguity = dense<[1, 4]> : tensor<2xi32>, tt.divisibility = dense<[1, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
      %mask: tensor<64x4xi1, #blocked>,
      %dst: !ttg.memdesc<64x4xf16, #shared, #smem, mutable>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x4xf16, #blocked>
    // expected-error@+2 {{cannot lower 'amdg.buffer_load_to_local' to a direct-to-LDS copy: the load can only be vectorized to 1 element(s) (16 bits), which is narrower than any direct-to-LDS transaction width supported on this target. The mask is the limiting factor: the pointers/offsets allow a vector of 4 element(s) but the mask only 1}}
    // expected-error@+1 {{failed to legalize operation 'amdg.buffer_load_to_local' that was explicitly marked illegal}}
    %0 = amdg.buffer_load_to_local %ptr[%offsets] mask=%mask other=%zero into %dst : !tt.ptr<f16>[tensor<64x4xi32, #blocked>] tensor<64x4xf16, #blocked> -> <64x4xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 1024 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @pointer_limits_direct_to_lds_width(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %offsets: tensor<64x4xi32, #blocked> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<[1, 16]> : tensor<2xi32>, tt.constancy = dense<[1, 1]> : tensor<2xi32>},
      %dst: !ttg.memdesc<64x4xf16, #shared, #smem, mutable>) {
    // expected-error@+2 {{cannot lower 'amdg.buffer_load_to_local' to a direct-to-LDS copy: the load can only be vectorized to 1 element(s) (16 bits), which is narrower than any direct-to-LDS transaction width supported on this target. Increase the provable alignment of the source}}
    // expected-error@+1 {{failed to legalize operation 'amdg.buffer_load_to_local' that was explicitly marked illegal}}
    %0 = amdg.buffer_load_to_local %ptr[%offsets] into %dst : !tt.ptr<f16>[tensor<64x4xi32, #blocked>] -> <64x4xf16, #shared, #smem, mutable>
    tt.return
  }
}
