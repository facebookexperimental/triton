// RUN: sed 's/opndA,smem,1,0/opndA,smem,two,0/' %S/ws_memory_planner_annotation.mlir | triton-opt --nvgpu-test-ws-memory-planner=num-buffers=2 -o /dev/null
// XFAIL: *

// Regression test for B-18-F3 / T273497320.
// Malformed numeric fields in user-provided tt.autows channel annotations
// should be rejected without letting std::stoi terminate the pass.
