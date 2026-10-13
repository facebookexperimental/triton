; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -o - %s | FileCheck %s --check-prefix=GFX9
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -o - %s | FileCheck %s --check-prefix=GFX9
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -o - %s | FileCheck %s --check-prefix=GFX9
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -o - %s | FileCheck %s --check-prefix=GFX1250

; The AMD TTGIR-to-LLVM tests verify that local and global barriers carry the
; corresponding amdgpu-synchronize-as MMRA tag, while mixed barriers carry no
; tag. Verify here that LLVM turns those fences into the target waits required
; before the hardware barrier.

declare void @llvm.amdgcn.s.barrier()

; GFX9-LABEL: local_barrier:
; GFX9:       s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  s_barrier
; GFX1250-LABEL: local_barrier:
; GFX1250:       s_wait_loadcnt_dscnt 0x0
; GFX1250-NEXT:  s_wait_kmcnt 0x0
; GFX1250-NOT:   s_wait_storecnt
; GFX1250:       s_barrier_signal -1
; GFX1250-NEXT:  s_barrier_wait -1
define void @local_barrier() {
  fence syncscope("workgroup") release, !mmra !0
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !0
  ret void
}

; GFX9-LABEL: global_barrier:
; GFX9:       s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  s_barrier
; GFX1250-LABEL: global_barrier:
; GFX1250:       s_wait_loadcnt_dscnt 0x0
; GFX1250-NEXT:  s_wait_kmcnt 0x0
; GFX1250-NEXT:  s_wait_storecnt 0x0
; GFX1250-NEXT:  s_barrier_signal -1
; GFX1250-NEXT:  s_barrier_wait -1
define void @global_barrier() {
  fence syncscope("workgroup") release, !mmra !1
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !1
  ret void
}

; GFX9-LABEL: mixed_barrier:
; GFX9:       s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  s_barrier
; GFX1250-LABEL: mixed_barrier:
; GFX1250:       s_wait_loadcnt_dscnt 0x0
; GFX1250-NEXT:  s_wait_kmcnt 0x0
; GFX1250-NEXT:  s_wait_storecnt 0x0
; GFX1250-NEXT:  s_barrier_signal -1
; GFX1250-NEXT:  s_barrier_wait -1
define void @mixed_barrier() {
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  ret void
}

!0 = !{!"amdgpu-synchronize-as", !"local"}
!1 = !{!"amdgpu-synchronize-as", !"global"}
