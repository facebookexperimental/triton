// RUN: triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=3 post-channel-creation=1" | FileCheck %s

// Test: Option 3 in-body SMEM-rotation for a MULTI-BUFFERED (numBuffers=3)
// subtiled-region SMEM reuse group, observed at the code-partition stage (the
// ttng.subtiled_region is still present, before lowering).
//
// For a multi-buffered subtiled reuse member, insertAsyncComm computes the
// staging-buffer slot AND the shared barrier bufferIdx/phase INSIDE the tile body
// from the op's builtin tileIdx:
//   flattened = accumCnt + tileIdx   (extui i32 -> i64)
//   bufferIdx = flattened % numBuffers(3)
//   view      = memdesc_index[base, bufferIdx]
// The numTiles stride lives on the loop-carried reuse-group counter (it advances
// by numTiles(2) per iteration), NOT in this index math -- so the flattened
// stream is iter*numTiles + tileIdx without an in-body `* numTiles`. accumCnt and
// the representative multibuffer alloc (3x128x64) are threaded as shared args;
// the producer local_store dest / consumer async_tma_copy source are rewired to
// the in-body view; and the now-dead per-tile buffer positions are removed
// (SubtiledRegionOp::removePerTilePosition). columnOffset / data-leaf positions
// stay per-tile.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @subtiled_smem_channel
  // CHECK: ttg.warp_specialize
  //
  // Counter strides (the fix): the subtiled reuse-group counter advances by
  // numTiles (=2) per iteration -- the numTiles factor lives on the counter, not
  // the in-body index math -- while the unique-channel counter advances by 1.
  // CHECK:      scf.for {{.*}}iter_args
  // CHECK:        arith.addi %{{[0-9a-z_]+}}, %c2_i64
  //
  // Producer region (epilogue, async_task_id 1): per_tile keeps ONLY the two
  // data leaves (the two f32 tensors) -- the SMEM buffer per-tile position was
  // removed and rewired to an in-body view. The base alloc (3x128x64) and
  // accumCnt are threaded as shared args.
  // CHECK:      ttng.subtiled_region per_tile({{.*}} : tensor<128x64xf32, #linear>, tensor<128x64xf32, #linear>) shared({{.*}}!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>{{.*}}) {async_task_id = array<i32: 1>, numTiles = 2 : i32}
  // In-body flattened count = (accumCnt + tileIdx) % numBuffers(3); accumCnt
  // already carries the numTiles stride, so there is NO in-body `* numTiles`.
  // Then the SAME %IDX indexes the staging view that local_store writes and the
  // producer barriers, proving data slot == barrier slot.
  // CHECK:      arith.extui
  // CHECK:      arith.addi %{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}
  // CHECK:      arith.divui %{{[0-9a-z_]+}}, %c3_i64
  // CHECK:      %[[IDX:[0-9]+]] = arith.trunci %{{[0-9a-z_]+}} {async_task_id = array<i32: 1>} : i64 to i32
  // CHECK:      %[[VIEW:[0-9]+]] = ttg.memdesc_index %{{[0-9a-z_]+}}[%[[IDX]]] {async_task_id = array<i32: 1>} : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  // CHECK:      nvws.producer_acquire %{{[0-9a-z_]+}}, %[[IDX]], %{{[0-9a-z_]+}}
  // CHECK:      ttg.local_store %{{[0-9a-z_]+}}, %[[VIEW]]
  // CHECK:      nvws.producer_commit %{{[0-9a-z_]+}}, %[[IDX]]
  // The unique-channel (non-subtile) counter advances by 1.
  // CHECK:      arith.addi %{{[0-9a-z_]+}}, %c1_i64
  //
  // Consumer region (epilogue_store, async_task_id 2): per_tile keeps ONLY the
  // two columnOffsets (i32, i32) -- both the dead leaf and the TMA-source buffer
  // positions were removed. Same in-body slot feeds the view / async_tma_copy.
  // CHECK:      ttng.subtiled_region per_tile({{.*}} : i32, i32) shared({{.*}}!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>{{.*}}) {async_task_id = array<i32: 2>, numTiles = 2 : i32}
  // CHECK:      arith.extui
  // CHECK:      arith.addi %{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}
  // CHECK:      arith.divui %{{[0-9a-z_]+}}, %c3_i64
  // CHECK:      %[[CIDX:[0-9]+]] = arith.trunci %{{[0-9a-z_]+}} {async_task_id = array<i32: 2>} : i64 to i32
  // CHECK:      %[[CVIEW:[0-9]+]] = ttg.memdesc_index %{{[0-9a-z_]+}}[%[[CIDX]]] {async_task_id = array<i32: 2>} : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  // CHECK:      nvws.consumer_wait %{{[0-9a-z_]+}}, %[[CIDX]], %{{[0-9a-z_]+}}
  // CHECK:      ttng.async_tma_copy_local_to_global %{{[0-9a-z_]+}}[%{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}] %[[CVIEW]]
  // CHECK:      nvws.consumer_release %{{[0-9a-z_]+}}, %[[CIDX]]
  tt.func @subtiled_smem_channel(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %off0: i32, %off1: i32, %off2: i32,
      %lhs: tensor<128x64xf32, #linear>,
      %rhs: tensor<128x64xf32, #linear>) {
    %smem0 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index

    scf.for %iv = %c0 to %c10 step %c1 {
      %dummy = arith.constant {async_task_id = array<i32: 0>} 0 : i32

      // Epilogue SubtiledRegionOp (task 1): truncf + local_store
      ttng.subtiled_region
          per_tile(%rhs, %lhs, %smem0, %smem1 :
                   tensor<128x64xf32, #linear>, tensor<128x64xf32, #linear>,
                   !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
                   !ttg.memdesc<128x64xf16, #shared, #smem, mutable>)
          {numTiles = 2 : i32, async_task_id = array<i32: 1>}
        tile(%t0: tensor<128x64xf32, #linear>,
             %t1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
             %tidx: i32) {
          %trunc = arith.truncf %t0 {async_task_id = array<i32: 1>}
            : tensor<128x64xf32, #linear> to tensor<128x64xf16, #linear>
          ttg.local_store %trunc, %t1 {async_task_id = array<i32: 1>}
            : tensor<128x64xf16, #linear> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.subtiled_region_yield
        }

      // TMA store SubtiledRegionOp (task 2): async_tma_copy
      ttng.subtiled_region
          per_tile(%smem0, %smem1, %off1, %off2 :
                   !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
                   !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
                   i32, i32)
          shared(%desc, %off0 :
                 !tt.tensordesc<tensor<128x64xf16, #shared>>, i32)
          {numTiles = 2 : i32, async_task_id = array<i32: 2>}
        tile(%t0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
             %t1: i32,
             %tdesc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
             %toff0: i32, %tidx: i32) {
          ttng.async_tma_copy_local_to_global %tdesc[%toff0, %t1] %t0
            {async_task_id = array<i32: 2>}
            : !tt.tensordesc<tensor<128x64xf16, #shared>>,
              !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.subtiled_region_yield
        }
    } {async_task_id = array<i32: 0, 1, 2>, tt.warp_specialize,
       tt.separate_epilogue_store = true,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["compute", "epilogue", "epilogue_store"]}

    tt.return
  }
}
