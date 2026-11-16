// RUN: triton-opt --tlx-print-ttgir-to-tlx %s -split-input-file --verify-diagnostics | FileCheck %s

// tl.make_tensor_descriptor on device lowers to a global scratch allocation, a
// ttng.tensormap_create writing the descriptor into it, and a
// reinterpret_tensor_descriptor reading it back. Printed through the generic
// op table these came out as tlx.allocate_tensor_descriptor() with no `num` and
// a positional tensormap argument dump. They map onto the TLX descriptor
// pointer API instead: tensormap dims are innermost-first with byte strides,
// and the box is clipped to the swizzle width, so the shape and strides are
// reversed and rescaled and the block shape comes from the reinterpreted type.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: def device_descriptor(
  // CHECK: [[PTR:[a-z_0-9]+]] = tlx.allocate_tensor_descriptor(num=1)
  // CHECK: tlx.make_tensor_descriptor([[PTR]], base, shape=[rows, 128], strides=[128, 1], block_shape=[128, 128])
  // CHECK: tl.extra.cuda.experimental_tensormap_fenceproxy_acquire([[PTR]])
  // CHECK: tlx.reinterpret_tensor_descriptor([[PTR]], block_shape=[128, 128], dtype=tl.bfloat16)
  tt.func public @device_descriptor(%base: !tt.ptr<bf16> loc("base"), %rows: i32 loc("rows")) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i64 = arith.constant 256 : i64
    %p = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
    ttng.tensormap_create %p, %base, [%c64_i32, %c128_i32], [%c128_i32, %rows], [%c256_i64], [%c1_i32, %c1_i32] {elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %p : !tt.ptr<i8>
    %d = ttng.reinterpret_tensor_descriptor %p : !tt.ptr<i8> to !tt.tensordesc<128x128xbf16, #shared>
    tt.return
  }

  // A runtime byte stride is rescaled in the generated code, several
  // descriptors share one allocation, and NaN padding is kept.
  // CHECK-LABEL: def runtime_stride_nan_padding(
  // CHECK: tlx.allocate_tensor_descriptor(num=2)
  // CHECK: tlx.make_tensor_descriptor({{.*}}, base, shape=[rows, 64], strides=[(stride // 2), 1], block_shape=[64, 64], padding_option="nan")
  tt.func public @runtime_stride_nan_padding(%base: !tt.ptr<bf16> loc("base"), %rows: i32 loc("rows"), %stride: i64 loc("stride")) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %p = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 256 : i32} : !tt.ptr<i8>
    ttng.tensormap_create %p, %base, [%c64_i32, %c64_i32], [%c64_i32, %rows], [%stride], [%c1_i32, %c1_i32] {elem_type = 10 : i32, fill_mode = 1 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    %d = ttng.reinterpret_tensor_descriptor %p : !tt.ptr<i8> to !tt.tensordesc<64x64xbf16, #shared>
    tt.return
  }

  // A descriptor pointer that reaches its reinterpret through another value
  // keeps the tensormap box as its block shape, which make_tensor_descriptor
  // turns back into the same tensormap.
  // CHECK-LABEL: def block_shape_from_box(
  // CHECK: tlx.make_tensor_descriptor({{.*}}, base, shape=[rows, 128], strides=[128, 1], block_shape=[128, 64])
  tt.func public @block_shape_from_box(%base: !tt.ptr<bf16> loc("base"), %rows: i32 loc("rows")) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i64 = arith.constant 256 : i64
    %p = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
    ttng.tensormap_create %p, %base, [%c64_i32, %c128_i32], [%c128_i32, %rows], [%c256_i64], [%c1_i32, %c1_i32] {elem_type = 10 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    tt.return
  }

  // Sub-byte elements have no whole-byte element stride to convert to; the
  // op is flagged instead of dividing by a zero byte width.
  // CHECK-LABEL: def sub_byte_elements(
  // CHECK: # unsupported: tensormap_create over i4
  tt.func public @sub_byte_elements(%base: !tt.ptr<i4> loc("base"), %rows: i32 loc("rows")) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i64 = arith.constant 64 : i64
    %p = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
    // expected-error @+1 {{tensormap_create over a sub-byte element type does not round-trip to TLX}}
    ttng.tensormap_create %p, %base, [%c128_i32, %c128_i32], [%c128_i32, %rows], [%c64_i64], [%c1_i32, %c1_i32] {elem_type = 0 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 0 : i32} : (!tt.ptr<i8>, !tt.ptr<i4>, i32, i32, i32, i32, i64, i32, i32) -> ()
    tt.return
  }
}
