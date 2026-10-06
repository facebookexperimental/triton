// RUN: triton-opt %s --tlx-resolve-placeholder-layouts | FileCheck %s --check-prefix=EARLY
// RUN: triton-opt %s --tlx-propagate-layout --tlx-resolve-placeholder-layouts --tritongpu-remove-layout-conversions --tlx-finalize-user-layouts | FileCheck %s --check-prefix=FINAL

// Register user-layout removal changes the concrete destination expected by
// reshape inference. Repair the reshape immediately so early pass verification
// succeeds, then let RLC move the required conversion outside restricted EXEC.

#src = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[32, 0], [64, 0], [16, 0]], block = []}>
#dst = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
#wrapped_dst = #tlx.no_verify_layout<#tlx.user_layout<#dst>>

module attributes {tlx.has_tlx_ops = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 32 : i32} {
  // EARLY-LABEL: tt.func @repair_register_user_layout_reshape
  // EARLY: ttg.warp_predicate
  // EARLY: %[[RESHAPED:.*]] = tt.reshape
  // EARLY-NEXT: %[[RESTORED:.*]] = ttg.require_layout %[[RESHAPED]]
  // EARLY: tt.store {{.*}}, %[[RESTORED]]
  // FINAL-LABEL: tt.func @repair_register_user_layout_reshape
  // FINAL: %[[COMPATIBLE_SRC:.*]] = ttg.convert_layout %{{.*}}
  // FINAL: ttg.warp_predicate
  // FINAL-NOT: ttg.convert_layout
  // FINAL: tt.reshape %[[COMPATIBLE_SRC]]
  tt.func @repair_register_user_layout_reshape(
      %predicate: i1, %src: tensor<128x128xf32, #src>,
      %ptrs: tensor<128x2x64x!tt.ptr<f32>, #wrapped_dst>) {
    ttg.warp_predicate %predicate () {
      %reshaped = tt.reshape %src : tensor<128x128xf32, #src> -> tensor<128x2x64xf32, #wrapped_dst>
      tt.store %ptrs, %reshaped : tensor<128x2x64x!tt.ptr<f32>, #wrapped_dst>
      ttg.predicate_yield
    } {wave_uniform} : (i1) -> ()
    tt.return
  }
}
