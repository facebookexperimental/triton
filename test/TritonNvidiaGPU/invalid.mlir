// RUN: triton-opt --split-input-file %s --verify-diagnostics

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @map_smem_to_remote(%arg: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    %c1_i32 = arith.constant 1 : i32
    // expected-error @+1 {{Invalid memory space for remote MemDesc}}
    %0 = ttng.map_to_remote_buffer %arg, %c1_i32: !ttg.memdesc<1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @alloc_tensor_memory() {
    // expected-error @+1 {{uninitialized alloc must have a mutable memdesc type}}
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @alloc_tensor_memory() {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %0 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    // expected-error @+1 {{Cannot store into an immutable alloc}}
    ttng.tmem_store %cst, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#scales = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#tmem = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @alloc_tensor_memory(%arg: !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>) {
    %cst = arith.constant dense<0> : tensor<128x4xi8, #scales>
    %0 = ttng.tmem_alloc %cst : (tensor<128x4xi8, #scales>) -> !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory>
    // expected-error @+1 {{Cannot copy into an immutable alloc}}
    ttng.tmem_copy %arg, %0 : !ttg.memdesc<128x4xi8, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_tma_gather(%desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                          %bar: !ttg.memdesc<2xi32, #shared1, #ttg.shared_memory, mutable>,
                          %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>,
                          %pred: i1) {
  // expected-error @below {{barrier allocation must be a descriptor of Nxi64 type with N <= number of CTAs}}
  ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<2xi32, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>, i1
  tt.return
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
tt.func @async_tma_gather(%desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                          %bar: !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>,
                          %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory>,
                          %pred: i1) {
  // expected-error @below {{cannot store into immutable memory}}
  ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory>, i1
  tt.return
}
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
tt.func @wgmma(%a: tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>, %b: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, %c: tensor<128x128xf16, #mma>) {
  // expected-error @below {{in-register LHS operand must have a kWidth of 2 but got 1}}
  %0 = ttng.warp_group_dot %a, %b, %c : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> -> tensor<128x128xf16, #mma>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<1x256x32xf32, #shared>>) -> tensor<256x32xf32, #blocked> {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // expected-error @below {{TMA descriptor must have NVMMA shared layout}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<tensor<1x256x32xf32, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<1x256x32xf32, #shared>>) -> tensor<256x32xf32, #blocked> {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{TMA descriptor layout must not be transposed}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<tensor<1x256x32xf32, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma32 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#nvmma64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared_mbar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<1x256x64xf32, #nvmma32>>) {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x64xf32, #nvmma64, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_mbar, #smem, mutable>
    // expected-error @below {{TMA descriptor layout must match shared layout}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<tensor<1x256x64xf32, #nvmma32>>, !ttg.memdesc<1xi64, #shared_mbar, #smem, mutable> -> !ttg.memdesc<256x64xf32, #nvmma64, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem_f16 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5(%a: !ttg.memdesc<128x128xbf16, #shared, #ttg.shared_memory>,
                  %b: !ttg.memdesc<128x256xbf16, #shared1, #ttg.shared_memory>,
                  %c: !ttg.memdesc<128x256xf16, #tmem_f16, #ttng.tensor_memory, mutable>,
                  %accUse: i1,
                  %pred: i1,
                  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                  %barrierPred: i1) {
    // expected-error @below {{unsupported accumulator dtype for operand types 'bf16' and 'bf16', accumulator dtype is 'f16' but must be one of ['f32']}}
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred, %barrier[%barrierPred] {is_async} :
       !ttg.memdesc<128x128xbf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xbf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf16, #tmem_f16, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

// Verify: tileMappings must have at least one tile
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_empty_tile_mappings(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %accum_cnt: i64) {
    // expected-error @+1 {{tileMappings must have at least one tile}}
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        accum_cnts(%accum_cnt : i64)
        tile_mappings = []
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0.0 : f32
        ttng.subtiled_region_yield %c0 : f32
      } tile(%arg0: f32) {
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: tileMappings inner array length must match tile block args
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_wrong_mapping_length(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %accum_cnt: i64) {
    // expected-error @+1 {{tileMappings[0] has 0 entries but tile region has 2 block arguments (expected 2 or 1)}}
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        accum_cnts(%accum_cnt : i64)
        tile_mappings = [array<i32>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32, %arg1: i32) {
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: setup index out of range
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_index_out_of_range(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %accum_cnt: i64) {
    // expected-error @+1 {{tileMappings[0][0] = 5 is out of range [0, 2)}}
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        accum_cnts(%accum_cnt : i64)
        tile_mappings = [array<i32: 5>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: type mismatch between setup output and tile block arg
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_type_mismatch(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %accum_cnt: i64) {
    // expected-error @+1 {{type mismatch: setup output 0 has type 'i32' but tile block arg 0 has type 'f32'}}
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        accum_cnts(%accum_cnt : i64)
        tile_mappings = [array<i32: 0>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: f32) {
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: barrierIdx out of range
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_barrier_idx_out_of_range(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %accum_cnt: i64) {
    // expected-error @+1 {{barrierAnnotations[0] has barrierIdx=3 but there are only 1 barriers}}
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        accum_cnts(%accum_cnt : i64)
        tile_mappings = [array<i32: 0>]
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 3, placement = after,
              targetOpIdx = 0, barrierOpKind = "arrive_barrier">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: wait_barrier without corresponding phase
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_wait_no_phase(
      %bar0: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %bar1: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %accum_cnt: i64) {
    // expected-error @+1 {{barrierAnnotations[0] is a wait_barrier with barrierIdx=1 but there are only 1 accumCnts}}
    ttng.subtiled_region
        barriers(%bar0, %bar1 : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        accum_cnts(%accum_cnt : i64)
        tile_mappings = [array<i32: 0>]
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 1, placement = before,
              targetOpIdx = 0, barrierOpKind = "wait_barrier">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: unknown barrierOpKind
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_unknown_barrier_kind(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %accum_cnt: i64) {
    // expected-error @+1 {{barrierAnnotations[0] has unknown barrierOpKind 'bogus'}}
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        accum_cnts(%accum_cnt : i64)
        tile_mappings = [array<i32: 0>]
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = after,
              targetOpIdx = 0, barrierOpKind = "bogus">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: targetOpIdx out of range
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_target_op_idx_out_of_range(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %accum_cnt: i64) {
    // expected-error @+1 {{barrierAnnotations[0] has targetOpIdx=5 but tile region has only 1 non-terminator ops}}
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        accum_cnts(%accum_cnt : i64)
        tile_mappings = [array<i32: 0>]
        barrier_annotations = [
          #ttng.barrier_annotation<barrierIdx = 0, placement = after,
              targetOpIdx = 5, barrierOpKind = "arrive_barrier">
        ]
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: i32) {
        %res = arith.addi %arg0, %arg0 : i32
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// Verify: teardown result count mismatch
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_teardown_result_mismatch(
      %bar: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
      %accum_cnt: i64) {
    // expected-error @+1 {{teardown yields 1 values but op has 0 results}}
    ttng.subtiled_region
        barriers(%bar : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>)
        accum_cnts(%accum_cnt : i64)
        tile_mappings = [array<i32: 0>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        ttng.subtiled_region_yield %c0 : i32
      } tile(%arg0: i32) {
        ttng.subtiled_region_yield
      } teardown {
        %c42 = arith.constant 42 : i32
        ttng.subtiled_region_yield %c42 : i32
      }
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @subtiled_region_interleaved_task_ids() {
    // expected-error @+1 {{tile body has interleaved async_task_id groups}}
    ttng.subtiled_region
        tile_mappings = [array<i32: 0>, array<i32: 1>]
        barrier_annotations = []
      setup {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        ttng.subtiled_region_yield %c0, %c1 : i32, i32
      } tile(%arg0: i32) {
        %a = arith.index_cast %arg0 {async_task_id = array<i32: 3>} : i32 to index
        %b = arith.index_cast %arg0 {async_task_id = array<i32: 4>} : i32 to index
        %c = arith.index_cast %arg0 {async_task_id = array<i32: 3>} : i32 to index
        ttng.subtiled_region_yield
      } teardown {
        ttng.subtiled_region_yield
      }
    tt.return
  }
}

// -----

// expected-error @+1 {{After removing the zero bases the layout must be bijective}}
#linear = #ttg.linear<{register = [[0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1]], warp = [[16, 0], [8, 0]], block = []}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @invalid_linear_layout(%arg0: tensor<32x64xi32, #linear>) {
    tt.return
  }
}

// -----

// Test that reduction with warps split across N dimension is rejected
// 128x256 with 8 warps -> warpsPerCTA = [4, 2] (2 warps in N)
#blocked_split = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked_red = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#tmem_warp_split = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_red_warp_split_rejected() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked_split>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x256xf32, #blocked_split>) -> !ttg.memdesc<128x256xf32, #tmem_warp_split, #ttng.tensor_memory, mutable>
    // expected-error @below {{tmem_load reduction with N dimension sharded across threads is not supported.}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>} : !ttg.memdesc<128x256xf32, #tmem_warp_split, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked_split>, tensor<128xf32, #blocked_red>
    tt.return
  }
}

// -----

// Test that reduction with N shared across threads is rejected
#blocked_split = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_red = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#bm64_bn128 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_red_16x32bx2_atom_rejected() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked_split>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<64x128xf32, #blocked_split>) -> !ttg.memdesc<64x128xf32, #bm64_bn128, #ttng.tensor_memory, mutable>
    // expected-error @below {{tmem_load reduction with N dimension sharded across threads is not supported.}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>} : !ttg.memdesc<64x128xf32, #bm64_bn128, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked_split>, tensor<64xf32, #blocked_red>
    tt.return
  }
}

// -----

// Test: abs requires redOp to be set
#blocked_abs = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_abs = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_abs_requires_redop() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked_abs>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked_abs>) -> !ttg.memdesc<128x128xf32, #tmem_abs, #ttng.tensor_memory, mutable>
    // expected-error @below {{'abs' requires 'redOp' to be set}}
    %result = ttng.tmem_load %0 {abs = true} : !ttg.memdesc<128x128xf32, #tmem_abs, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_abs>
    tt.return
  }
}

// -----

// Test: NaN requires redOp to be set
#blocked_nan = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_nan = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_nan_requires_redop() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked_nan>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked_nan>) -> !ttg.memdesc<128x128xf32, #tmem_nan, #ttng.tensor_memory, mutable>
    // expected-error @below {{'NaN' requires 'redOp' to be set}}
    %result = ttng.tmem_load %0 {NaN = true} : !ttg.memdesc<128x128xf32, #tmem_nan, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_nan>
    tt.return
  }
}

// -----

// Test: abs requires f32 element type
#blocked_abs_i32 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_red_abs_i32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem_abs_i32 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_abs_requires_f32() {
    %cst_0 = arith.constant dense<0> : tensor<128x128xi32, #blocked_abs_i32>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xi32, #blocked_abs_i32>) -> !ttg.memdesc<128x128xi32, #tmem_abs_i32, #ttng.tensor_memory, mutable>
    // expected-error @below {{'abs' requires floating-point element type (f32)}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>, abs = true} : !ttg.memdesc<128x128xi32, #tmem_abs_i32, #ttng.tensor_memory, mutable> -> tensor<128x128xi32, #blocked_abs_i32>, tensor<128xi32, #blocked_red_abs_i32>
    tt.return
  }
}

// -----

// Test: NaN requires f32 element type
#blocked_nan_i32 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_red_nan_i32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem_nan_i32 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_nan_requires_f32() {
    %cst_0 = arith.constant dense<0> : tensor<128x128xi32, #blocked_nan_i32>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xi32, #blocked_nan_i32>) -> !ttg.memdesc<128x128xi32, #tmem_nan_i32, #ttng.tensor_memory, mutable>
    // expected-error @below {{'NaN' requires floating-point element type (f32)}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>, NaN = true} : !ttg.memdesc<128x128xi32, #tmem_nan_i32, #ttng.tensor_memory, mutable> -> tensor<128x128xi32, #blocked_nan_i32>, tensor<128xi32, #blocked_red_nan_i32>
    tt.return
  }
}
