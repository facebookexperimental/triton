// RUN: triton-opt -split-input-file --convert-triton-to-tritongpu="target=hip:gfx950 num-warps=8 threads-per-warp=64 num-ctas=1" %s | FileCheck %s

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [8, 1], instrShape = [32, 32, 16], isTransposed = true}>
#result = #tlx.no_verify_layout<#tlx.user_layout<#mma>>
#operand_a = #tlx.no_verify_layout<#tlx.user_layout<#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>>
#operand_b = #tlx.no_verify_layout<#tlx.user_layout<#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @pinned_dot
  // CHECK-NOT: #ttg.blocked
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.dot {{.*}} -> tensor<256x64xf32, #{{.*}}>
  tt.func @pinned_dot(%a: tensor<256x64xbf16, #operand_a>,
                      %b: tensor<64x64xbf16, #operand_b>,
                      %c: tensor<256x64xf32, #result>)
      -> tensor<256x64xf32, #result> {
    %dot = tt.dot %a, %b, %c : tensor<256x64xbf16, #operand_a> * tensor<64x64xbf16, #operand_b> -> tensor<256x64xf32, #result>
    tt.return %dot : tensor<256x64xf32, #result>
  }
}
