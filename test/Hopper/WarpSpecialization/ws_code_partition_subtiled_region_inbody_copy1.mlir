// RUN: TRITON_USE_META_WS=1 triton-opt %s --nvgpu-test-ws-code-partition="num-buffers=3 post-channel-creation=1" | FileCheck %s

// Test: Option 3 in-body SMEM-rotation for a SINGLE-BUFFERED (buffer.copy = 1)
// both-endpoints-subtiled SMEM reuse group -- the DP=1 epilogue shape that OOM'd
// (addmm test_autows_addmm_tma_persistent[...-True-1-True-...-2-...-256-...]).
//
// The two per-tile staging allocs %smem0/%smem1 collapse into ONE ChannelPost
// (cross-task producer region task 1, consumer region task 2). At buffer.copy=1
// the collapse must still fire: the representative is wrapped to a 1x128x64
// array, the in-body slot math runs with numBuffers=1 (bufferIdx == 0, phase
// alternates), and the now-dead sibling alloc %smem1 is ERASED. That last step
// is the OOM fix -- without it %smem1 survives as a second 16 KiB staging buffer.
//
//   flattened = accumCnt + tileIdx        (extui i32 -> i64)
//   bufferIdx = flattened % numBuffers(1) == 0
//   view      = memdesc_index[base(1x128x64), bufferIdx]
// The numTiles(2) stride lives on the loop-carried reuse-group counter, NOT in
// the index math. accumCnt and the 1x128x64 base alloc are threaded as shared
// args; producer local_store dest / consumer async_tma_copy source are rewired
// to the in-body view; dead per-tile buffer positions are removed.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @subtiled_smem_channel_copy1
  //
  // Exactly ONE physical staging buffer survives: the representative wrapped to
  // a 1x128x64 array (emitted at function entry, before ttg.warp_specialize).
  // The sibling per-tile alloc was collapsed away (the OOM fix) -- no bare
  // 128x64 mutable #shared staging local_alloc remains before warp_specialize.
  // CHECK:      ttg.local_alloc {{.*}}-> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
  // CHECK-NOT:  ttg.local_alloc{{.*}}-> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  // CHECK:      ttg.warp_specialize
  //
  // Counter stride (the collapse still carries the numTiles stride at copy=1):
  // the subtiled reuse-group counter advances by numTiles(=2) per iteration.
  // CHECK:      scf.for {{.*}}iter_args
  // CHECK:        arith.addi %{{[0-9a-z_]+}}, %c2_i64
  //
  // Producer region (epilogue, ttg.partition 1): per_tile keeps ONLY the two
  // data leaves; the SMEM per-tile position was removed and rewired to an
  // in-body view into the 1x128x64 base (threaded as a shared arg).
  // CHECK:      ttng.subtiled_region per_tile({{.*}} : tensor<128x64xf32, #linear>, tensor<128x64xf32, #linear>) shared({{.*}}!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>{{.*}}) {numTiles = 2 : i32, ttg.partition = array<i32: 1>}
  // In-body flattened count = (accumCnt + tileIdx) % numBuffers(1); the divui by
  // %c1_i64 is the numBuffers=1 slot math (bufferIdx == 0). The SAME %IDX indexes
  // the staging view and the producer barriers (data slot == barrier slot).
  // CHECK:      arith.extui
  // CHECK:      arith.addi %{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}
  // CHECK:      arith.divui %{{[0-9a-z_]+}}, %c1_i64
  // CHECK:      %[[IDX:[0-9]+]] = arith.trunci %{{[0-9a-z_]+}} {ttg.partition = array<i32: 1>} : i64 to i32
  // CHECK:      %[[VIEW:[0-9]+]] = ttg.memdesc_index %{{[0-9a-z_]+}}[%[[IDX]]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  // CHECK:      nvws.producer_acquire %{{[0-9a-z_]+}}, %[[IDX]], %{{[0-9a-z_]+}}
  // CHECK:      ttg.local_store %{{[0-9a-z_]+}}, %[[VIEW]]
  // CHECK:      nvws.producer_commit %{{[0-9a-z_]+}}, %[[IDX]]
  //
  // Consumer region (epilogue_store, ttg.partition 2): per_tile keeps ONLY the
  // two columnOffsets (i32, i32). Same in-body slot feeds the view / TMA copy.
  // CHECK:      ttng.subtiled_region per_tile({{.*}} : i32, i32) shared({{.*}}!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>{{.*}}) {numTiles = 2 : i32, ttg.partition = array<i32: 2>}
  // CHECK:      arith.extui
  // CHECK:      arith.addi %{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}
  // CHECK:      arith.divui %{{[0-9a-z_]+}}, %c1_i64
  // CHECK:      %[[CIDX:[0-9]+]] = arith.trunci %{{[0-9a-z_]+}} {ttg.partition = array<i32: 2>} : i64 to i32
  // CHECK:      %[[CVIEW:[0-9]+]] = ttg.memdesc_index %{{[0-9a-z_]+}}[%[[CIDX]]] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
  // CHECK:      nvws.consumer_wait %{{[0-9a-z_]+}}, %[[CIDX]], %{{[0-9a-z_]+}}
  // CHECK:      ttng.async_tma_copy_local_to_global %{{[0-9a-z_]+}}[%{{[0-9a-z_]+}}, %{{[0-9a-z_]+}}] %[[CVIEW]]
  // CHECK:      nvws.consumer_release %{{[0-9a-z_]+}}, %[[CIDX]]
  tt.func @subtiled_smem_channel_copy1(
      %desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %off0: i32, %off1: i32, %off2: i32,
      %lhs: tensor<128x64xf32, #linear>,
      %rhs: tensor<128x64xf32, #linear>) {
    %smem0 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index

    scf.for %iv = %c0 to %c10 step %c1 {
      %dummy = arith.constant {ttg.partition = array<i32: 0>} 0 : i32

      // Epilogue SubtiledRegionOp (task 1): truncf + local_store
      ttng.subtiled_region
          per_tile(%rhs, %lhs, %smem0, %smem1 :
                   tensor<128x64xf32, #linear>, tensor<128x64xf32, #linear>,
                   !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
                   !ttg.memdesc<128x64xf16, #shared, #smem, mutable>)
          {numTiles = 2 : i32, ttg.partition = array<i32: 1>}
        tile(%t0: tensor<128x64xf32, #linear>,
             %t1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
             %tidx: i32) {
          %trunc = arith.truncf %t0 {ttg.partition = array<i32: 1>}
            : tensor<128x64xf32, #linear> to tensor<128x64xf16, #linear>
          ttg.local_store %trunc, %t1 {ttg.partition = array<i32: 1>}
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
          {numTiles = 2 : i32, ttg.partition = array<i32: 2>}
        tile(%t0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
             %t1: i32,
             %tdesc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
             %toff0: i32, %tidx: i32) {
          ttng.async_tma_copy_local_to_global %tdesc[%toff0, %t1] %t0
            {ttg.partition = array<i32: 2>}
            : !tt.tensordesc<tensor<128x64xf16, #shared>>,
              !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          ttng.subtiled_region_yield
        }
    } {ttg.partition = array<i32: 0, 1, 2>, tt.warp_specialize,
       tt.separate_epilogue_store = true,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["compute", "epilogue", "epilogue_store"]}

    tt.return
  }
}
