// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=TAGGED,COMMON
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=TAGGED,COMMON
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=GENERIC,COMMON
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=RDNA4,COMMON
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=TAGGED,COMMON
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1100 cu-mode=true" --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=TAGGED,COMMON
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1200 cu-mode=true" --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=TAGGED,COMMON

// RDNA CTA barriers use tagged lowering only in CU mode, where waves share one L0.
// COMMON-DAG: [[$LOCAL_MMRA_TAG:#[A-Za-z0-9_]+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">
// COMMON-DAG: [[$GLOBAL_MMRA_TAG:#[A-Za-z0-9_]+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"global">

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // TAGGED-LABEL: llvm.func @lower_barrier
  // GENERIC-LABEL: llvm.func @lower_barrier
  // RDNA4-LABEL: llvm.func @lower_barrier
  tt.func @lower_barrier() {
    // TAGGED: llvm.fence syncscope("workgroup") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // TAGGED-NEXT: rocdl.s.barrier
    // TAGGED-NEXT: llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}

    // GENERIC: llvm.fence syncscope("workgroup") release{{$}}
    // GENERIC-NEXT: rocdl.s.barrier
    // GENERIC-NEXT: llvm.fence syncscope("workgroup") acquire{{$}}

    // RDNA4: llvm.fence syncscope("workgroup") release{{$}}
    // RDNA4-NEXT: rocdl.s.barrier.signal id = -1
    // RDNA4-NEXT: rocdl.s.barrier.wait id = -1
    // RDNA4-NEXT: llvm.fence syncscope("workgroup") acquire{{$}}
    ttg.barrier local
    tt.return
  }

  // COMMON-LABEL: llvm.func @lower_warp_barrier
  tt.func @lower_warp_barrier() {
    // COMMON-NOT: rocdl.s.barrier
    // COMMON: llvm.fence syncscope("wavefront") release {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // COMMON-NEXT: rocdl.wave.barrier
    // COMMON-NEXT: llvm.fence syncscope("wavefront") acquire {llvm.mmra = [[$LOCAL_MMRA_TAG]]}
    // COMMON-NOT: rocdl.s.barrier
    // COMMON: llvm.return
    ttg.barrier warp local
    tt.return
  }

  // GPU scope takes the agent-scope path on every target, including RDNA in
  // WGP mode, whose CTA barriers fall back to the generic lowering above.
  // COMMON-LABEL: llvm.func @lower_gpu_barrier
  tt.func @lower_gpu_barrier() {
    // COMMON-NOT: syncscope("workgroup")
    // COMMON: llvm.fence syncscope("agent") release{{$}}
    // COMMON-NEXT: rocdl.s.barrier{{$}}
    // COMMON-NEXT: llvm.fence syncscope("agent") acquire{{$}}
    // COMMON-NOT: llvm.fence
    // COMMON: llvm.return
    ttg.barrier gpu all
    tt.return
  }

  // A global-only GPU barrier is tagged so the legalizer can skip LDS waits.
  // COMMON-LABEL: llvm.func @lower_gpu_global_barrier
  tt.func @lower_gpu_global_barrier() {
    // COMMON-NOT: syncscope("workgroup")
    // COMMON: llvm.fence syncscope("agent") release {llvm.mmra = [[$GLOBAL_MMRA_TAG]]}
    // COMMON-NEXT: rocdl.s.barrier{{$}}
    // COMMON-NEXT: llvm.fence syncscope("agent") acquire {llvm.mmra = [[$GLOBAL_MMRA_TAG]]}
    // COMMON-NOT: llvm.fence
    // COMMON: llvm.return
    ttg.barrier gpu global_read|global_write
    tt.return
  }
}
