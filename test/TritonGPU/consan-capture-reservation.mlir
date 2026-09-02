// RUN: split-file %s %t
// RUN: not triton-opt %t/missing.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/missing.mlir --check-prefix=MISSING
// RUN: not triton-opt %t/too-small.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/too-small.mlir --check-prefix=SMALL

//--- missing.mlir

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @missing_reservation() {
    // MISSING: WarpSpecialize op is missing 'consan.extra_capture_bytes'
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

//--- too-small.mlir

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @small_reservation() {
    // SMALL: ConSan WarpSpecialize capture reservation is too small: reserved 0 bytes, but 1 captures require 8 bytes
    ttg.warp_specialize() attributes {consan.extra_capture_bytes = 0 : i32}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

//--- convert-only.mlir

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#dst = #ttg.slice<{dim = 1, parent = #dst_parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @convert_only_reservation(%arg0: tensor<128xi32, #src>) {
    %0 = ttg.convert_layout %arg0 : tensor<128xi32, #src> -> tensor<128xi32, #dst>
    // CONVERT: ttg.warp_specialize() attributes {consan.extra_capture_bytes = 24 : i32}
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

//--- empty-ws.mlir

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @empty_ws_reservation() {
    // The lock capture makes WarpSpecialize scratch a shared-memory effect,
    // so the stable reservation is lock + write/read visibility pointers.
    // EMPTY: ttg.warp_specialize() attributes {consan.extra_capture_bytes = 24 : i32}
    // EMPTY-INTEGRATION: ttg.warp_specialize({{.*}}allocation.size = 24 : i32
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

//--- scratch-missing-offset.mlir

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#dst = #ttg.slice<{dim = 1, parent = #dst_parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @scratch_missing_offset(%arg0: tensor<128xi32, #src>) {
    // SCRATCH-MISSING-OFFSET: error: compiler scratch metadata requires integer allocation.offset and allocation.size attributes
    %0 = ttg.convert_layout %arg0 {allocation.size = 512 : i32} : tensor<128xi32, #src> -> tensor<128xi32, #dst>
    tt.return
  }
}

//--- scratch-invalid.mlir

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#dst = #ttg.slice<{dim = 1, parent = #dst_parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @scratch_invalid(%arg0: tensor<128xi32, #src>) {
    // SCRATCH-INVALID: error: invalid compiler scratch allocation metadata: offset 16777215, size 2
    %0 = ttg.convert_layout %arg0 {allocation.offset = 16777215 : i32, allocation.size = 2 : i32} : tensor<128xi32, #src> -> tensor<128xi32, #dst>
    tt.return
  }
}
