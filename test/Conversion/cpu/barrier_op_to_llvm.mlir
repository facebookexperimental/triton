// RUN: triton-opt %s -triton-cpu-debug-ops-to-llvm | FileCheck %s

// Each program runs on one CPU thread, so the rendezvous is dropped. Programs
// run concurrently, so a GPU-scope barrier keeps a system-scope fence.

// CHECK-LABEL: tt.func @lower_barriers
tt.func @lower_barriers() {
  // CHECK-NOT: ttg.barrier
  // CHECK-NOT: llvm.fence
  ttg.barrier all
  // CHECK: llvm.fence acq_rel{{$}}
  // CHECK-NOT: ttg.barrier
  // CHECK-NOT: llvm.fence
  ttg.barrier gpu global_read|global_write
  // CHECK: tt.return
  tt.return
}
