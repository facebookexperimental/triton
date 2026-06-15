// RUN: triton-opt %s --nvgpu-test-ws-memory-planner=num-buffers=2 -o /dev/null

// Regression test for B-18-F3 / T273497320.
// Malformed numeric fields in user-provided tt.autows channel annotations
// should be rejected without letting std::stoi terminate the pass.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @malformed_numeric_annotation() attributes {noinline = false} {
    %c0_i32 = arith.constant {tt.autows = "{\22channels\22: [\22opndA,smem,two,0\22]}"} 0 : i32
    tt.return
  }
}
