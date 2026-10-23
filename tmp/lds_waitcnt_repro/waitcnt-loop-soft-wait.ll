; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -o - %s | FileCheck %s

; Reduced from the gfx950 warp-pipelined addmm (addmm_warppipe.py).
; The ds_read issued at the end of each iteration is still pending at the next
; iteration's local barrier, so the release fence must wait for it. The
; preheader has no pending LDS access, which is what triggers the bug.
; SIInsertWaitcnts erases the fence's soft lgkmcnt(0) on its first visit to the
; loop header, before the backedge state is known.
; Fails before llvm/llvm-project 437f88f61b (#220357), passes after it.

; CHECK-LABEL: loop_carried_ds_read:
; CHECK:       .LBB0_[[LOOP:[0-9]+]]:
; CHECK-NOT:   s_barrier
; CHECK:       s_waitcnt lgkmcnt(0)
; CHECK-NEXT:  s_barrier
; CHECK:       ds_read
; CHECK:       s_cbranch_{{.*}} .LBB0_[[LOOP]]
define amdgpu_kernel void @loop_carried_ds_read(ptr addrspace(3) %lds, ptr addrspace(1) %out, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi <4 x float> [ zeroinitializer, %entry ], [ %acc.next, %loop ]
  %b = phi <8 x bfloat> [ zeroinitializer, %entry ], [ %b.next, %loop ]
  fence syncscope("workgroup") release, !mmra !0
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !0
  %acc.next = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> zeroinitializer, <8 x bfloat> %b, <4 x float> %acc, i32 0, i32 0, i32 0)
  %b.next = load <8 x bfloat>, ptr addrspace(3) %lds, align 16
  %i.next = add i32 %i, 1
  %cont = icmp slt i32 %i.next, %n
  br i1 %cont, label %loop, label %exit

exit:
  store <4 x float> %acc.next, ptr addrspace(1) %out, align 16
  ret void
}

declare void @llvm.amdgcn.s.barrier()
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)

!0 = !{!"amdgpu-synchronize-as", !"local"}
