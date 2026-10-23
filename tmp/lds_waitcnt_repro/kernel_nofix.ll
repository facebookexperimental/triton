; ModuleID = 'addmm_warppipe_kernel'
source_filename = "addmm_warppipe_kernel"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: nounwind
define amdgpu_kernel void @addmm_warppipe_kernel(ptr addrspace(1) inreg nofree readonly captures(none) %0, ptr addrspace(1) inreg nofree readonly captures(none) %1, ptr addrspace(1) inreg nofree readonly captures(none) %2, ptr addrspace(1) inreg nofree writeonly captures(none) %3, ptr addrspace(1) inreg nofree readnone captures(none) %4, ptr addrspace(1) inreg nofree readnone captures(none) %5) local_unnamed_addr #0 {
  %7 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %8 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %7)
  %9 = lshr i32 %8, 6
  %10 = and i32 %9, 7
  %11 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %12 = icmp slt i32 %11, 1312
  br i1 %12, label %13, label %22

13:                                               ; preds = %6
  %14 = sdiv i32 %11, 8
  %15 = mul i32 %14, 8
  %.decomposed = sub i32 %11, %15
  %16 = sdiv i32 %11, 32
  %17 = shl nsw i32 %16, 5
  %18 = shl nsw i32 %.decomposed, 2
  %19 = srem i32 %14, 4
  %20 = add nsw i32 %18, %19
  %21 = add i32 %20, %17
  br label %22

22:                                               ; preds = %13, %6
  %23 = phi i32 [ %21, %13 ], [ %11, %6 ]
  %.frozen = freeze i32 %23
  %24 = sdiv i32 %.frozen, 328
  %25 = shl nsw i32 %24, 3
  %26 = sub nsw i32 32, %25
  %27 = tail call i32 @llvm.smin.i32(i32 %26, i32 8)
  %28 = srem i32 %23, %27
  %29 = add nsw i32 %25, %28
  %30 = mul i32 %24, 328
  %.decomposed207 = sub i32 %.frozen, %30
  %31 = sdiv i32 %.decomposed207, %27
  %32 = icmp sgt i32 %29, -1
  tail call void @llvm.assume(i1 %32)
  %33 = icmp sgt i32 %31, -1
  tail call void @llvm.assume(i1 %33)
  %34 = shl i32 %29, 6
  %35 = and i32 %7, 63
  %36 = shl nuw nsw i32 %10, 6
  %37 = or disjoint i32 %36, %35
  %38 = and i32 %7, 56
  %39 = lshr exact i32 %38, 1
  %40 = and i32 %36, 192
  %41 = lshr exact i32 %40, 6
  %42 = and i32 %36, 256
  %43 = lshr exact i32 %42, 3
  %44 = or disjoint i32 %41, %43
  %45 = or disjoint i32 %44, %39
  %46 = and i32 %7, 7
  %47 = shl nuw nsw i32 %46, 3
  %48 = or disjoint i32 %34, %45
  %49 = srem i32 %48, 2033
  %50 = shl nuw nsw i32 %31, 6
  %51 = or disjoint i32 %50, %45
  %.lhs.trunc = trunc nuw nsw i32 %51 to i16
  %52 = urem i16 %.lhs.trunc, 2577
  %.zext = zext nneg i16 %52 to i32
  %53 = mul nsw i32 %49, 2560
  %54 = or disjoint i32 %53, %47
  %55 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %1, i16 0, i64 2147483646, i32 159744)
  %56 = mul nuw nsw i32 %10, 1056
  %57 = getelementptr i8, ptr addrspace(3) @global_smem, i32 %56
  %58 = shl nsw i32 %54, 1
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %55, ptr addrspace(3) %57, i32 16, i32 %58, i32 0, i32 0, i32 0), !alias.scope !4
  %59 = mul nuw nsw i32 %.zext, 2560
  %60 = or disjoint i32 %59, %47
  %61 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %2, i16 0, i64 2147483646, i32 159744)
  %62 = getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 25312), i32 %56
  %63 = shl nuw nsw i32 %60, 1
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %61, ptr addrspace(3) %62, i32 16, i32 %63, i32 0, i32 0, i32 0), !alias.scope !4
  tail call void @llvm.amdgcn.asyncmark()
  %64 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8448), i32 %56
  %65 = or disjoint i32 %58, 128
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %55, ptr addrspace(3) %64, i32 16, i32 %65, i32 0, i32 0, i32 0), !alias.scope !4
  %66 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 33760), i32 %56
  %67 = or disjoint i32 %63, 128
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %61, ptr addrspace(3) %66, i32 16, i32 %67, i32 0, i32 0, i32 0), !alias.scope !4
  tail call void @llvm.amdgcn.asyncmark()
  %68 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 16896), i32 %56
  %69 = or disjoint i32 %58, 256
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %55, ptr addrspace(3) %68, i32 16, i32 %69, i32 0, i32 0, i32 0), !alias.scope !4
  %70 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 42208), i32 %56
  %71 = or disjoint i32 %63, 256
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %61, ptr addrspace(3) %70, i32 16, i32 %71, i32 0, i32 0, i32 0), !alias.scope !4
  tail call void @llvm.amdgcn.asyncmark()
  tail call void @llvm.amdgcn.wait.asyncmark(i16 1)
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  %72 = and i32 %7, 3
  %73 = shl nuw nsw i32 %72, 10
  %74 = and i32 %7, 12
  %75 = shl nuw nsw i32 %74, 5
  %76 = and i32 %7, 48
  %77 = shl nuw nsw i32 %42, 1
  %78 = shl nuw nsw i32 %72, 5
  %79 = or disjoint i32 %77, %75
  %80 = or disjoint i32 %79, %76
  %81 = or disjoint i32 %80, %73
  %82 = add nuw nsw i32 %81, %78
  %83 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %82
  %84 = load <8 x bfloat>, ptr addrspace(3) %83, align 16
  %85 = add nuw nsw i32 %82, 64
  %86 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %85
  %87 = load <8 x bfloat>, ptr addrspace(3) %86, align 16
  %88 = add nuw nsw i32 %82, 4224
  %89 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %88
  %90 = load <8 x bfloat>, ptr addrspace(3) %89, align 16
  %91 = add nuw nsw i32 %82, 4288
  %92 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %91
  %93 = load <8 x bfloat>, ptr addrspace(3) %92, align 16
  %94 = shl nuw nsw i32 %37, 5
  %95 = and i32 %94, 4480
  %96 = and i32 %36, 64
  %97 = shl nuw nsw i32 %96, 3
  %98 = or disjoint i32 %95, %97
  %99 = or disjoint i32 %98, %76
  %100 = or disjoint i32 %99, %73
  %101 = lshr i32 %100, 5
  %102 = and i32 %101, 67108832
  %103 = add nuw nsw i32 %102, %100
  %104 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 25312), i32 %103
  %105 = load <8 x bfloat>, ptr addrspace(3) %104, align 16
  %106 = add nuw nsw i32 %103, 64
  %107 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 25312), i32 %106
  %108 = load <8 x bfloat>, ptr addrspace(3) %107, align 16
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  %109 = icmp samesign ult i32 %7, 256
  br i1 %109, label %111, label %110

110:                                              ; preds = %22
  tail call void @llvm.amdgcn.s.barrier()
  br label %111

111:                                              ; preds = %110, %22
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %112 = shufflevector <8 x bfloat> %84, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %113 = shufflevector <8 x bfloat> %84, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %114 = shufflevector <8 x bfloat> %84, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %115 = shufflevector <8 x bfloat> %84, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %116 = shufflevector <8 x bfloat> %87, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %117 = shufflevector <8 x bfloat> %87, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %118 = shufflevector <8 x bfloat> %87, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %119 = shufflevector <8 x bfloat> %87, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %120 = shufflevector <8 x bfloat> %90, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %121 = shufflevector <8 x bfloat> %90, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %122 = shufflevector <8 x bfloat> %90, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %123 = shufflevector <8 x bfloat> %90, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %124 = shufflevector <8 x bfloat> %93, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %125 = shufflevector <8 x bfloat> %93, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %126 = shufflevector <8 x bfloat> %93, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %127 = shufflevector <8 x bfloat> %93, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %128 = shufflevector <8 x bfloat> %105, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %129 = shufflevector <8 x bfloat> %105, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %130 = shufflevector <8 x bfloat> %105, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %131 = shufflevector <8 x bfloat> %105, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %132 = shufflevector <8 x bfloat> %108, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %133 = shufflevector <8 x bfloat> %108, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %134 = shufflevector <8 x bfloat> %108, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %135 = shufflevector <8 x bfloat> %108, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  br label %136

136:                                              ; preds = %237, %111
  %137 = phi float [ 0.000000e+00, %111 ], [ %276, %237 ]
  %138 = phi float [ 0.000000e+00, %111 ], [ %277, %237 ]
  %139 = phi float [ 0.000000e+00, %111 ], [ %278, %237 ]
  %140 = phi float [ 0.000000e+00, %111 ], [ %279, %237 ]
  %141 = phi float [ 0.000000e+00, %111 ], [ %282, %237 ]
  %142 = phi float [ 0.000000e+00, %111 ], [ %283, %237 ]
  %143 = phi float [ 0.000000e+00, %111 ], [ %284, %237 ]
  %144 = phi float [ 0.000000e+00, %111 ], [ %285, %237 ]
  %145 = phi i32 [ 0, %111 ], [ %287, %237 ]
  %146 = phi <2 x bfloat> [ %112, %111 ], [ %311, %237 ]
  %147 = phi <2 x bfloat> [ %113, %111 ], [ %312, %237 ]
  %148 = phi <2 x bfloat> [ %114, %111 ], [ %313, %237 ]
  %149 = phi <2 x bfloat> [ %115, %111 ], [ %314, %237 ]
  %150 = phi <2 x bfloat> [ %116, %111 ], [ %315, %237 ]
  %151 = phi <2 x bfloat> [ %117, %111 ], [ %316, %237 ]
  %152 = phi <2 x bfloat> [ %118, %111 ], [ %317, %237 ]
  %153 = phi <2 x bfloat> [ %119, %111 ], [ %318, %237 ]
  %154 = phi <2 x bfloat> [ %120, %111 ], [ %319, %237 ]
  %155 = phi <2 x bfloat> [ %121, %111 ], [ %320, %237 ]
  %156 = phi <2 x bfloat> [ %122, %111 ], [ %321, %237 ]
  %157 = phi <2 x bfloat> [ %123, %111 ], [ %322, %237 ]
  %158 = phi <2 x bfloat> [ %124, %111 ], [ %323, %237 ]
  %159 = phi <2 x bfloat> [ %125, %111 ], [ %324, %237 ]
  %160 = phi <2 x bfloat> [ %126, %111 ], [ %325, %237 ]
  %161 = phi <2 x bfloat> [ %127, %111 ], [ %326, %237 ]
  %162 = phi <2 x bfloat> [ %128, %111 ], [ %327, %237 ]
  %163 = phi <2 x bfloat> [ %129, %111 ], [ %328, %237 ]
  %164 = phi <2 x bfloat> [ %130, %111 ], [ %329, %237 ]
  %165 = phi <2 x bfloat> [ %131, %111 ], [ %330, %237 ]
  %166 = phi <2 x bfloat> [ %132, %111 ], [ %331, %237 ]
  %167 = phi <2 x bfloat> [ %133, %111 ], [ %332, %237 ]
  %168 = phi <2 x bfloat> [ %134, %111 ], [ %333, %237 ]
  %169 = phi <2 x bfloat> [ %135, %111 ], [ %334, %237 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.wait.asyncmark(i16 1)
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %170 = shufflevector <2 x bfloat> %146, <2 x bfloat> %147, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %171 = shufflevector <2 x bfloat> %148, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %172 = shufflevector <8 x bfloat> %170, <8 x bfloat> %171, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %173 = shufflevector <2 x bfloat> %149, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %174 = shufflevector <8 x bfloat> %172, <8 x bfloat> %173, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %175 = shufflevector <2 x bfloat> %150, <2 x bfloat> %151, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %176 = shufflevector <2 x bfloat> %152, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %177 = shufflevector <8 x bfloat> %175, <8 x bfloat> %176, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %178 = shufflevector <2 x bfloat> %153, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %179 = shufflevector <8 x bfloat> %177, <8 x bfloat> %178, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %180 = shufflevector <2 x bfloat> %154, <2 x bfloat> %155, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %181 = shufflevector <2 x bfloat> %156, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %182 = shufflevector <8 x bfloat> %180, <8 x bfloat> %181, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %183 = shufflevector <2 x bfloat> %157, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %184 = shufflevector <8 x bfloat> %182, <8 x bfloat> %183, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %185 = shufflevector <2 x bfloat> %158, <2 x bfloat> %159, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %186 = shufflevector <2 x bfloat> %160, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %187 = shufflevector <8 x bfloat> %185, <8 x bfloat> %186, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %188 = shufflevector <2 x bfloat> %161, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %189 = shufflevector <8 x bfloat> %187, <8 x bfloat> %188, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %190 = shufflevector <2 x bfloat> %162, <2 x bfloat> %163, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %191 = shufflevector <2 x bfloat> %164, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %192 = shufflevector <8 x bfloat> %190, <8 x bfloat> %191, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %193 = shufflevector <2 x bfloat> %165, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %194 = shufflevector <8 x bfloat> %192, <8 x bfloat> %193, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %195 = shufflevector <2 x bfloat> %166, <2 x bfloat> %167, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %196 = shufflevector <2 x bfloat> %168, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %197 = shufflevector <8 x bfloat> %195, <8 x bfloat> %196, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %198 = shufflevector <2 x bfloat> %169, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %199 = shufflevector <8 x bfloat> %197, <8 x bfloat> %198, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %200 = insertelement <4 x float> poison, float %137, i64 0
  %201 = insertelement <4 x float> %200, float %138, i64 1
  %202 = insertelement <4 x float> %201, float %139, i64 2
  %203 = insertelement <4 x float> %202, float %140, i64 3
  %204 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %194, <8 x bfloat> %174, <4 x float> %203, i32 0, i32 0, i32 0)
  %205 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %199, <8 x bfloat> %179, <4 x float> %204, i32 0, i32 0, i32 0)
  %206 = insertelement <4 x float> poison, float %141, i64 0
  %207 = insertelement <4 x float> %206, float %142, i64 1
  %208 = insertelement <4 x float> %207, float %143, i64 2
  %209 = insertelement <4 x float> %208, float %144, i64 3
  %210 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %194, <8 x bfloat> %184, <4 x float> %209, i32 0, i32 0, i32 0)
  %211 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %199, <8 x bfloat> %189, <4 x float> %210, i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.barrier()
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %.lhs.trunc118 = trunc nuw i32 %145 to i8
  %212 = urem i8 %.lhs.trunc118, 3
  %.zext119 = zext nneg i8 %212 to i32
  %213 = or disjoint i32 %145, 1
  %.lhs.trunc120 = trunc nuw i32 %213 to i8
  %214 = urem i8 %.lhs.trunc120, 3
  %.zext121 = zext nneg i8 %214 to i32
  %215 = shl nuw nsw i32 %145, 6
  %216 = add nuw nsw i32 %215, 192
  %217 = add nsw i32 %216, %54
  %218 = mul nuw nsw i32 %.zext119, 4224
  %gep = getelementptr [2 x i8], ptr addrspace(3) %57, i32 %218
  %219 = shl i32 %217, 1
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %55, ptr addrspace(3) %gep, i32 16, i32 %219, i32 0, i32 0, i32 0), !alias.scope !4
  %220 = add nuw nsw i32 %216, %60
  %gep123 = getelementptr [2 x i8], ptr addrspace(3) %62, i32 %218
  %221 = shl nuw nsw i32 %220, 1
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %61, ptr addrspace(3) %gep123, i32 16, i32 %221, i32 0, i32 0, i32 0), !alias.scope !4
  tail call void @llvm.amdgcn.asyncmark()
  %222 = mul nuw nsw i32 %.zext121, 4224
  %223 = getelementptr [2 x i8], ptr addrspace(3) @global_smem, i32 %222
  %224 = getelementptr inbounds nuw i8, ptr addrspace(3) %223, i32 %82
  %225 = load <8 x bfloat>, ptr addrspace(3) %224, align 16
  %226 = getelementptr inbounds nuw i8, ptr addrspace(3) %223, i32 %85
  %227 = load <8 x bfloat>, ptr addrspace(3) %226, align 16
  %228 = getelementptr inbounds nuw i8, ptr addrspace(3) %223, i32 %88
  %229 = load <8 x bfloat>, ptr addrspace(3) %228, align 16
  %230 = getelementptr inbounds nuw i8, ptr addrspace(3) %223, i32 %91
  %231 = load <8 x bfloat>, ptr addrspace(3) %230, align 16
  %232 = getelementptr [2 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 25312), i32 %222
  %233 = getelementptr inbounds nuw i8, ptr addrspace(3) %232, i32 %103
  %234 = load <8 x bfloat>, ptr addrspace(3) %233, align 16
  %235 = getelementptr inbounds nuw i8, ptr addrspace(3) %232, i32 %106
  %236 = load <8 x bfloat>, ptr addrspace(3) %235, align 16
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %exitcond.not = icmp eq i32 %145, 36
  br i1 %exitcond.not, label %335, label %237

237:                                              ; preds = %136
  %238 = shufflevector <8 x bfloat> %236, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %239 = shufflevector <8 x bfloat> %236, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %240 = shufflevector <8 x bfloat> %234, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %241 = shufflevector <8 x bfloat> %234, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %242 = shufflevector <8 x bfloat> %231, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %243 = shufflevector <8 x bfloat> %231, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %244 = shufflevector <8 x bfloat> %229, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %245 = shufflevector <8 x bfloat> %229, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %246 = shufflevector <8 x bfloat> %227, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %247 = shufflevector <8 x bfloat> %227, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %248 = shufflevector <8 x bfloat> %225, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %249 = shufflevector <8 x bfloat> %225, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.wait.asyncmark(i16 1)
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %250 = shufflevector <2 x bfloat> %249, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %251 = shufflevector <8 x bfloat> %225, <8 x bfloat> %250, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %252 = shufflevector <2 x bfloat> %248, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %253 = shufflevector <8 x bfloat> %251, <8 x bfloat> %252, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %254 = shufflevector <2 x bfloat> %247, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %255 = shufflevector <8 x bfloat> %227, <8 x bfloat> %254, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %256 = shufflevector <2 x bfloat> %246, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %257 = shufflevector <8 x bfloat> %255, <8 x bfloat> %256, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %258 = shufflevector <2 x bfloat> %245, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %259 = shufflevector <8 x bfloat> %229, <8 x bfloat> %258, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %260 = shufflevector <2 x bfloat> %244, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %261 = shufflevector <8 x bfloat> %259, <8 x bfloat> %260, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %262 = shufflevector <2 x bfloat> %243, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %263 = shufflevector <8 x bfloat> %231, <8 x bfloat> %262, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %264 = shufflevector <2 x bfloat> %242, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %265 = shufflevector <8 x bfloat> %263, <8 x bfloat> %264, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %266 = shufflevector <2 x bfloat> %241, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %267 = shufflevector <8 x bfloat> %234, <8 x bfloat> %266, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %268 = shufflevector <2 x bfloat> %240, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %269 = shufflevector <8 x bfloat> %267, <8 x bfloat> %268, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %270 = shufflevector <2 x bfloat> %239, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %271 = shufflevector <8 x bfloat> %236, <8 x bfloat> %270, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %272 = shufflevector <2 x bfloat> %238, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %273 = shufflevector <8 x bfloat> %271, <8 x bfloat> %272, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %274 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %269, <8 x bfloat> %253, <4 x float> %205, i32 0, i32 0, i32 0)
  %275 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %273, <8 x bfloat> %257, <4 x float> %274, i32 0, i32 0, i32 0)
  %276 = extractelement <4 x float> %275, i64 0
  %277 = extractelement <4 x float> %275, i64 1
  %278 = extractelement <4 x float> %275, i64 2
  %279 = extractelement <4 x float> %275, i64 3
  %280 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %269, <8 x bfloat> %261, <4 x float> %211, i32 0, i32 0, i32 0)
  %281 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %273, <8 x bfloat> %265, <4 x float> %280, i32 0, i32 0, i32 0)
  %282 = extractelement <4 x float> %281, i64 0
  %283 = extractelement <4 x float> %281, i64 1
  %284 = extractelement <4 x float> %281, i64 2
  %285 = extractelement <4 x float> %281, i64 3
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.barrier()
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %.lhs.trunc118.1 = trunc nuw i32 %213 to i8
  %286 = urem i8 %.lhs.trunc118.1, 3
  %.zext119.1 = zext nneg i8 %286 to i32
  %287 = add nuw nsw i32 %145, 2
  %.lhs.trunc120.1 = trunc nuw i32 %287 to i8
  %288 = urem i8 %.lhs.trunc120.1, 3
  %.zext121.1 = zext nneg i8 %288 to i32
  %289 = shl nuw nsw i32 %213, 6
  %290 = add nuw nsw i32 %289, 192
  %291 = add nsw i32 %290, %54
  %292 = mul nuw nsw i32 %.zext119.1, 4224
  %gep.1 = getelementptr [2 x i8], ptr addrspace(3) %57, i32 %292
  %293 = shl i32 %291, 1
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %55, ptr addrspace(3) %gep.1, i32 16, i32 %293, i32 0, i32 0, i32 0), !alias.scope !4
  %294 = add nuw nsw i32 %290, %60
  %gep123.1 = getelementptr [2 x i8], ptr addrspace(3) %62, i32 %292
  %295 = shl nuw nsw i32 %294, 1
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %61, ptr addrspace(3) %gep123.1, i32 16, i32 %295, i32 0, i32 0, i32 0), !alias.scope !4
  tail call void @llvm.amdgcn.asyncmark()
  %296 = mul nuw nsw i32 %.zext121.1, 4224
  %297 = getelementptr [2 x i8], ptr addrspace(3) @global_smem, i32 %296
  %298 = getelementptr inbounds nuw i8, ptr addrspace(3) %297, i32 %82
  %299 = load <8 x bfloat>, ptr addrspace(3) %298, align 16
  %300 = getelementptr inbounds nuw i8, ptr addrspace(3) %297, i32 %85
  %301 = load <8 x bfloat>, ptr addrspace(3) %300, align 16
  %302 = getelementptr inbounds nuw i8, ptr addrspace(3) %297, i32 %88
  %303 = load <8 x bfloat>, ptr addrspace(3) %302, align 16
  %304 = getelementptr inbounds nuw i8, ptr addrspace(3) %297, i32 %91
  %305 = load <8 x bfloat>, ptr addrspace(3) %304, align 16
  %306 = getelementptr [2 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 25312), i32 %296
  %307 = getelementptr inbounds nuw i8, ptr addrspace(3) %306, i32 %103
  %308 = load <8 x bfloat>, ptr addrspace(3) %307, align 16
  %309 = getelementptr inbounds nuw i8, ptr addrspace(3) %306, i32 %106
  %310 = load <8 x bfloat>, ptr addrspace(3) %309, align 16
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  %311 = shufflevector <8 x bfloat> %299, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %312 = shufflevector <8 x bfloat> %299, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %313 = shufflevector <8 x bfloat> %299, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %314 = shufflevector <8 x bfloat> %299, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %315 = shufflevector <8 x bfloat> %301, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %316 = shufflevector <8 x bfloat> %301, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %317 = shufflevector <8 x bfloat> %301, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %318 = shufflevector <8 x bfloat> %301, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %319 = shufflevector <8 x bfloat> %303, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %320 = shufflevector <8 x bfloat> %303, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %321 = shufflevector <8 x bfloat> %303, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %322 = shufflevector <8 x bfloat> %303, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %323 = shufflevector <8 x bfloat> %305, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %324 = shufflevector <8 x bfloat> %305, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %325 = shufflevector <8 x bfloat> %305, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %326 = shufflevector <8 x bfloat> %305, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %327 = shufflevector <8 x bfloat> %308, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %328 = shufflevector <8 x bfloat> %308, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %329 = shufflevector <8 x bfloat> %308, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %330 = shufflevector <8 x bfloat> %308, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %331 = shufflevector <8 x bfloat> %310, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %332 = shufflevector <8 x bfloat> %310, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %333 = shufflevector <8 x bfloat> %310, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %334 = shufflevector <8 x bfloat> %310, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  br label %136

335:                                              ; preds = %136
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  br i1 %109, label %336, label %337

336:                                              ; preds = %335
  tail call void @llvm.amdgcn.s.barrier()
  br label %337

337:                                              ; preds = %336, %335
  %338 = or disjoint i32 %10, 56
  %339 = or disjoint i32 %34, %338
  %340 = icmp slt i32 %339, 2033
  %341 = or disjoint i32 %50, %35
  %342 = icmp samesign ult i32 %341, 2577
  %343 = and i1 %342, %340
  %344 = or disjoint i32 %10, 48
  %345 = or disjoint i32 %34, %344
  %346 = icmp slt i32 %345, 2033
  %347 = and i1 %342, %346
  %348 = or disjoint i32 %10, 40
  %349 = or disjoint i32 %34, %348
  %350 = icmp slt i32 %349, 2033
  %351 = and i1 %342, %350
  %352 = or disjoint i32 %10, 32
  %353 = or disjoint i32 %34, %352
  %354 = icmp slt i32 %353, 2033
  %355 = and i1 %342, %354
  %356 = or disjoint i32 %10, 24
  %357 = or disjoint i32 %34, %356
  %358 = icmp slt i32 %357, 2033
  %359 = and i1 %342, %358
  %360 = or disjoint i32 %10, 16
  %361 = or disjoint i32 %34, %360
  %362 = icmp slt i32 %361, 2033
  %363 = and i1 %342, %362
  %364 = or disjoint i32 %10, 8
  %365 = or disjoint i32 %34, %364
  %366 = icmp slt i32 %365, 2033
  %367 = and i1 %342, %366
  %368 = or disjoint i32 %34, %10
  %369 = icmp slt i32 %368, 2033
  %370 = and i1 %342, %369
  %371 = lshr i32 %37, 3
  %372 = or disjoint i32 %34, %371
  %373 = icmp slt i32 %372, 2033
  %374 = or disjoint i32 %47, %50
  %375 = or disjoint i32 %374, 7
  %376 = icmp samesign ult i32 %375, 2577
  %377 = and i1 %376, %373
  %378 = or disjoint i32 %374, 6
  %379 = icmp samesign ult i32 %378, 2577
  %380 = and i1 %379, %373
  %381 = or disjoint i32 %374, 5
  %382 = icmp samesign ult i32 %381, 2577
  %383 = and i1 %382, %373
  %384 = or disjoint i32 %374, 4
  %385 = icmp samesign ult i32 %384, 2577
  %386 = and i1 %385, %373
  %387 = or disjoint i32 %374, 3
  %388 = icmp samesign ult i32 %387, 2577
  %389 = and i1 %388, %373
  %390 = or disjoint i32 %374, 2
  %391 = icmp samesign ult i32 %390, 2577
  %392 = and i1 %391, %373
  %393 = or disjoint i32 %374, 1
  %394 = icmp samesign ult i32 %393, 2577
  %395 = and i1 %394, %373
  %396 = icmp samesign ult i32 %374, 2577
  %397 = and i1 %396, %373
  %398 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %234, <8 x bfloat> %225, <4 x float> %205, i32 0, i32 0, i32 0)
  %399 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %236, <8 x bfloat> %227, <4 x float> %398, i32 0, i32 0, i32 0)
  %400 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %234, <8 x bfloat> %229, <4 x float> %211, i32 0, i32 0, i32 0)
  %401 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %236, <8 x bfloat> %231, <4 x float> %400, i32 0, i32 0, i32 0)
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0)
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  %402 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 16896), i32 %82
  %403 = load <8 x bfloat>, ptr addrspace(3) %402, align 16
  %404 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 16896), i32 %85
  %405 = load <8 x bfloat>, ptr addrspace(3) %404, align 16
  %406 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 16896), i32 %88
  %407 = load <8 x bfloat>, ptr addrspace(3) %406, align 16
  %408 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 16896), i32 %91
  %409 = load <8 x bfloat>, ptr addrspace(3) %408, align 16
  %410 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 42208), i32 %103
  %411 = load <8 x bfloat>, ptr addrspace(3) %410, align 16
  %412 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 42208), i32 %106
  %413 = load <8 x bfloat>, ptr addrspace(3) %412, align 16
  %414 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %411, <8 x bfloat> %403, <4 x float> %399, i32 0, i32 0, i32 0)
  %415 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %413, <8 x bfloat> %405, <4 x float> %414, i32 0, i32 0, i32 0)
  %416 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %411, <8 x bfloat> %407, <4 x float> %401, i32 0, i32 0, i32 0)
  %417 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %413, <8 x bfloat> %409, <4 x float> %416, i32 0, i32 0, i32 0)
  %418 = load <8 x bfloat>, ptr addrspace(3) %83, align 16
  %419 = load <8 x bfloat>, ptr addrspace(3) %86, align 16
  %420 = load <8 x bfloat>, ptr addrspace(3) %89, align 16
  %421 = load <8 x bfloat>, ptr addrspace(3) %92, align 16
  %422 = load <8 x bfloat>, ptr addrspace(3) %104, align 16
  %423 = load <8 x bfloat>, ptr addrspace(3) %107, align 16
  %424 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %422, <8 x bfloat> %418, <4 x float> %415, i32 0, i32 0, i32 0)
  %425 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %423, <8 x bfloat> %419, <4 x float> %424, i32 0, i32 0, i32 0)
  %426 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %422, <8 x bfloat> %420, <4 x float> %417, i32 0, i32 0, i32 0)
  %427 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %423, <8 x bfloat> %421, <4 x float> %426, i32 0, i32 0, i32 0)
  %428 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %55, i32 -2147483648, i32 0, i32 0)
  %429 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %61, i32 -2147483648, i32 0, i32 0)
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  %430 = shl nuw nsw i32 %46, 5
  %431 = and i32 %36, 384
  %432 = shl nuw nsw i32 %431, 2
  %433 = and i32 %7, 8
  %434 = shl nuw nsw i32 %433, 1
  %435 = and i32 %7, 16
  %436 = icmp eq i32 %435, 0
  %437 = select i1 %436, i32 0, i32 2112
  %438 = and i32 %7, 32
  %439 = shl nuw nsw i32 %438, 3
  %440 = icmp eq i32 %96, 0
  %441 = select i1 %440, i32 0, i32 4224
  %442 = or disjoint i32 %432, %430
  %443 = or disjoint i32 %441, %437
  %444 = xor i32 %443, %442
  %445 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %434
  %446 = getelementptr inbounds nuw i8, ptr addrspace(3) %445, i32 %444
  %447 = getelementptr inbounds nuw i8, ptr addrspace(3) %446, i32 %439
  store <4 x i32> %428, ptr addrspace(3) %447, align 16
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  %448 = shl nuw nsw i32 %37, 1
  %449 = and i32 %448, 608
  %450 = shl nuw nsw i32 %7, 4
  %451 = and i32 %450, 16
  %452 = and i32 %7, 2
  %453 = icmp eq i32 %452, 0
  %454 = select i1 %453, i32 0, i32 2112
  %455 = shl nuw nsw i32 %7, 6
  %456 = and i32 %455, 256
  %457 = icmp eq i32 %433, 0
  %458 = select i1 %457, i32 0, i32 4224
  %459 = xor i32 %449, %454
  %460 = or disjoint i32 %458, %459
  %461 = or disjoint i32 %460, %456
  %462 = or disjoint i32 %461, %451
  %463 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %462
  %464 = load <8 x bfloat>, ptr addrspace(3) %463, align 16
  %465 = getelementptr inbounds nuw i8, ptr addrspace(3) %463, i32 1024
  %466 = load <8 x bfloat>, ptr addrspace(3) %465, align 16
  %467 = xor i32 %462, 128
  %468 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %467
  %469 = load <8 x bfloat>, ptr addrspace(3) %468, align 16
  %470 = getelementptr inbounds nuw i8, ptr addrspace(3) %468, i32 1024
  %471 = load <8 x bfloat>, ptr addrspace(3) %470, align 16
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  %472 = and i32 %7, 6
  %473 = shl nuw nsw i32 %472, 10
  %474 = shl nuw nsw i32 %472, 5
  %475 = shl nuw nsw i32 %37, 2
  %476 = and i32 %475, 1632
  %477 = shl nuw nsw i32 %96, 1
  %478 = xor i32 %474, %477
  %479 = xor i32 %478, %476
  %480 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %479
  %481 = getelementptr inbounds nuw i8, ptr addrspace(3) %480, i32 %473
  %482 = getelementptr inbounds nuw i8, ptr addrspace(3) %481, i32 %451
  %483 = getelementptr inbounds nuw i8, ptr addrspace(3) %482, i32 %439
  store <4 x i32> %429, ptr addrspace(3) %483, align 16
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  %484 = shl nuw nsw i32 %40, 3
  %485 = shl nuw nsw i32 %433, 4
  %486 = icmp eq i32 %438, 0
  %487 = select i1 %486, i32 0, i32 2112
  %488 = or disjoint i32 %484, %78
  %489 = xor i32 %488, %487
  %490 = or disjoint i32 %489, %485
  %491 = or disjoint i32 %490, %456
  %492 = or disjoint i32 %491, %435
  %493 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %492
  %494 = load <8 x bfloat>, ptr addrspace(3) %493, align 16
  %495 = xor i32 %492, 4224
  %496 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %495
  %497 = load <8 x bfloat>, ptr addrspace(3) %496, align 16
  %498 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %494, <8 x bfloat> %464, <4 x float> %425, i32 0, i32 0, i32 0)
  %499 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %497, <8 x bfloat> %469, <4 x float> %498, i32 0, i32 0, i32 0)
  %500 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %494, <8 x bfloat> %466, <4 x float> %427, i32 0, i32 0, i32 0)
  %501 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %497, <8 x bfloat> %471, <4 x float> %500, i32 0, i32 0, i32 0)
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  %502 = shl nuw nsw i32 %7, 10
  %503 = and i32 %502, 15360
  %504 = shl nuw nsw i32 %74, 4
  %505 = shl nuw nsw i32 %72, 3
  %506 = and i32 %448, 224
  %507 = shl nuw nsw i32 %10, 8
  %508 = and i32 %507, 512
  %509 = or disjoint i32 %504, %505
  %510 = xor i32 %506, %509
  %511 = or disjoint i32 %508, %503
  %512 = or disjoint i32 %511, %510
  %513 = or disjoint i32 %512, %42
  %514 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %513
  %515 = shufflevector <4 x float> %499, <4 x float> %501, <2 x i32> <i32 0, i32 4>
  store <2 x float> %515, ptr addrspace(3) %514, align 8
  %516 = xor i32 %513, 8
  %517 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %516
  %518 = shufflevector <4 x float> %499, <4 x float> %501, <2 x i32> <i32 1, i32 5>
  store <2 x float> %518, ptr addrspace(3) %517, align 8
  %519 = xor i32 %513, 16
  %520 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %519
  %521 = shufflevector <4 x float> %499, <4 x float> %501, <2 x i32> <i32 2, i32 6>
  store <2 x float> %521, ptr addrspace(3) %520, align 8
  %522 = xor i32 %513, 24
  %523 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %522
  %524 = shufflevector <4 x float> %499, <4 x float> %501, <2 x i32> <i32 3, i32 7>
  store <2 x float> %524, ptr addrspace(3) %523, align 8
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  %525 = shl nuw nsw i32 %37, 4
  %526 = and i32 %525, 3584
  %527 = shl nuw nsw i32 %7, 3
  %528 = and i32 %527, 248
  %529 = lshr exact i32 %40, 3
  %530 = icmp eq i32 %42, 0
  %531 = select i1 %530, i32 0, i32 4160
  %532 = or disjoint i32 %526, %528
  %533 = or disjoint i32 %531, %529
  %534 = xor i32 %532, %533
  %535 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %534
  %536 = load <2 x float>, ptr addrspace(3) %535, align 8
  %537 = getelementptr inbounds nuw i8, ptr addrspace(3) %535, i32 256
  %538 = load <2 x float>, ptr addrspace(3) %537, align 8
  %539 = xor i32 %534, 8320
  %540 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %539
  %541 = load <2 x float>, ptr addrspace(3) %540, align 8
  %542 = getelementptr inbounds nuw i8, ptr addrspace(3) %540, i32 256
  %543 = load <2 x float>, ptr addrspace(3) %542, align 8
  %544 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %0, i16 0, i64 2147483646, i32 159744)
  %545 = shl nuw nsw i32 %374, 1
  %546 = select i1 %397, i32 %545, i32 -2147483648
  %547 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %544, i32 %546, i32 0, i32 0)
  %548 = shl nuw nsw i32 %393, 1
  %549 = select i1 %395, i32 %548, i32 -2147483648
  %550 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %544, i32 %549, i32 0, i32 0)
  %551 = shl nuw nsw i32 %390, 1
  %552 = select i1 %392, i32 %551, i32 -2147483648
  %553 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %544, i32 %552, i32 0, i32 0)
  %554 = shl nuw nsw i32 %387, 1
  %555 = select i1 %389, i32 %554, i32 -2147483648
  %556 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %544, i32 %555, i32 0, i32 0)
  %557 = shl nuw nsw i32 %384, 1
  %558 = select i1 %386, i32 %557, i32 -2147483648
  %559 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %544, i32 %558, i32 0, i32 0)
  %560 = shl nuw nsw i32 %381, 1
  %561 = select i1 %383, i32 %560, i32 -2147483648
  %562 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %544, i32 %561, i32 0, i32 0)
  %563 = shl nuw nsw i32 %378, 1
  %564 = select i1 %380, i32 %563, i32 -2147483648
  %565 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %544, i32 %564, i32 0, i32 0)
  %566 = shl nuw nsw i32 %375, 1
  %567 = select i1 %377, i32 %566, i32 -2147483648
  %568 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %544, i32 %567, i32 0, i32 0)
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  %569 = shl nuw nsw i32 %38, 7
  %570 = shl nuw nsw i32 %431, 1
  %571 = lshr exact i32 %96, 5
  %572 = or disjoint i32 %570, %571
  %573 = or disjoint i32 %572, %569
  %574 = or disjoint i32 %573, %430
  %575 = or disjoint i32 %574, %39
  %576 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %575
  store i16 %547, ptr addrspace(3) %576, align 2
  %577 = xor i32 %575, 4
  %578 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %577
  store i16 %550, ptr addrspace(3) %578, align 2
  %579 = xor i32 %575, 8
  %580 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %579
  store i16 %553, ptr addrspace(3) %580, align 2
  %581 = xor i32 %575, 12
  %582 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %581
  store i16 %556, ptr addrspace(3) %582, align 2
  %583 = xor i32 %575, 16
  %584 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %583
  store i16 %559, ptr addrspace(3) %584, align 2
  %585 = xor i32 %575, 20
  %586 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %585
  store i16 %562, ptr addrspace(3) %586, align 2
  %587 = xor i32 %575, 24
  %588 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %587
  store i16 %565, ptr addrspace(3) %588, align 2
  %589 = xor i32 %575, 28
  %590 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %589
  store i16 %568, ptr addrspace(3) %590, align 2
  fence syncscope("workgroup") release, !mmra !7
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire, !mmra !7
  %591 = shl nuw nsw i32 %10, 10
  %592 = shl nuw nsw i32 %35, 2
  %593 = shl nuw nsw i32 %10, 2
  %594 = or disjoint i32 %591, %592
  %595 = xor i32 %594, %593
  %596 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %595
  %597 = load <2 x i16>, ptr addrspace(3) %596, align 4
  %598 = getelementptr inbounds nuw i8, ptr addrspace(3) %596, i32 256
  %599 = load <2 x i16>, ptr addrspace(3) %598, align 4
  %600 = getelementptr inbounds nuw i8, ptr addrspace(3) %596, i32 512
  %601 = load <2 x i16>, ptr addrspace(3) %600, align 4
  %602 = getelementptr inbounds nuw i8, ptr addrspace(3) %596, i32 768
  %603 = load <2 x i16>, ptr addrspace(3) %602, align 4
  %604 = mul i32 %29, 164928
  %605 = mul nuw nsw i32 %10, 2577
  %606 = mul nuw nsw i32 %364, 2577
  %607 = mul nuw nsw i32 %360, 2577
  %608 = mul nuw nsw i32 %356, 2577
  %609 = mul nuw nsw i32 %352, 2577
  %610 = mul nuw nsw i32 %348, 2577
  %611 = mul nuw nsw i32 %344, 2577
  %612 = mul nuw nsw i32 %338, 2577
  %613 = add i32 %604, %50
  %614 = add i32 %613, %605
  %615 = add i32 %614, %35
  %616 = add i32 %613, %606
  %617 = add i32 %616, %35
  %618 = add i32 %613, %607
  %619 = add i32 %618, %35
  %620 = add i32 %613, %608
  %621 = add i32 %620, %35
  %622 = add i32 %613, %609
  %623 = add i32 %622, %35
  %624 = add i32 %613, %610
  %625 = add i32 %624, %35
  %626 = add i32 %613, %611
  %627 = add i32 %626, %35
  %628 = add i32 %613, %612
  %629 = add i32 %628, %35
  %630 = zext <2 x i16> %597 to <2 x i32>
  %631 = shl nuw <2 x i32> %630, splat (i32 16)
  %632 = bitcast <2 x i32> %631 to <2 x float>
  %633 = shufflevector <2 x float> %536, <2 x float> %541, <2 x i32> <i32 0, i32 2>
  %634 = fadd <2 x float> %633, %632
  %635 = zext <2 x i16> %599 to <2 x i32>
  %636 = shl nuw <2 x i32> %635, splat (i32 16)
  %637 = bitcast <2 x i32> %636 to <2 x float>
  %638 = shufflevector <2 x float> %538, <2 x float> %543, <2 x i32> <i32 0, i32 2>
  %639 = fadd <2 x float> %638, %637
  %640 = zext <2 x i16> %601 to <2 x i32>
  %641 = shl nuw <2 x i32> %640, splat (i32 16)
  %642 = bitcast <2 x i32> %641 to <2 x float>
  %643 = shufflevector <2 x float> %536, <2 x float> %541, <2 x i32> <i32 1, i32 3>
  %644 = fadd <2 x float> %643, %642
  %645 = zext <2 x i16> %603 to <2 x i32>
  %646 = shl nuw <2 x i32> %645, splat (i32 16)
  %647 = bitcast <2 x i32> %646 to <2 x float>
  %648 = shufflevector <2 x float> %538, <2 x float> %543, <2 x i32> <i32 1, i32 3>
  %649 = fadd <2 x float> %648, %647
  %650 = fptrunc <2 x float> %634 to <2 x bfloat>
  %651 = fptrunc <2 x float> %639 to <2 x bfloat>
  %652 = fptrunc <2 x float> %644 to <2 x bfloat>
  %653 = fptrunc <2 x float> %649 to <2 x bfloat>
  %654 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %3, i16 0, i64 2147483646, i32 159744)
  %.bc = bitcast <2 x bfloat> %650 to <2 x i16>
  %.extract = extractelement <2 x i16> %.bc, i64 0
  %655 = shl i32 %615, 1
  %656 = select i1 %370, i32 %655, i32 -2147483648
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract, ptr addrspace(8) %654, i32 %656, i32 0, i32 0)
  %.extract105 = extractelement <2 x i16> %.bc, i64 1
  %657 = shl i32 %617, 1
  %658 = select i1 %367, i32 %657, i32 -2147483648
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract105, ptr addrspace(8) %654, i32 %658, i32 0, i32 0)
  %.bc106 = bitcast <2 x bfloat> %651 to <2 x i16>
  %.extract107 = extractelement <2 x i16> %.bc106, i64 0
  %659 = shl i32 %619, 1
  %660 = select i1 %363, i32 %659, i32 -2147483648
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract107, ptr addrspace(8) %654, i32 %660, i32 0, i32 0)
  %.extract109 = extractelement <2 x i16> %.bc106, i64 1
  %661 = shl i32 %621, 1
  %662 = select i1 %359, i32 %661, i32 -2147483648
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract109, ptr addrspace(8) %654, i32 %662, i32 0, i32 0)
  %.bc110 = bitcast <2 x bfloat> %652 to <2 x i16>
  %.extract111 = extractelement <2 x i16> %.bc110, i64 0
  %663 = shl i32 %623, 1
  %664 = select i1 %355, i32 %663, i32 -2147483648
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract111, ptr addrspace(8) %654, i32 %664, i32 0, i32 0)
  %.extract113 = extractelement <2 x i16> %.bc110, i64 1
  %665 = shl i32 %625, 1
  %666 = select i1 %351, i32 %665, i32 -2147483648
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract113, ptr addrspace(8) %654, i32 %666, i32 0, i32 0)
  %.bc114 = bitcast <2 x bfloat> %653 to <2 x i16>
  %.extract115 = extractelement <2 x i16> %.bc114, i64 0
  %667 = shl i32 %627, 1
  %668 = select i1 %347, i32 %667, i32 -2147483648
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract115, ptr addrspace(8) %654, i32 %668, i32 0, i32 0)
  %.extract117 = extractelement <2 x i16> %.bc114, i64 1
  %669 = shl i32 %629, 1
  %670 = select i1 %343, i32 %669, i32 -2147483648
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %.extract117, ptr addrspace(8) %654, i32 %670, i32 0, i32 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) readnone, i16, i64, i32) #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32, i32, i32 immarg, i32 immarg) #5

; Function Attrs: nounwind
declare void @llvm.amdgcn.asyncmark() #6

; Function Attrs: nounwind
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #7

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.setprio(i16 immarg) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #7

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.waitcnt(i32 immarg) #9

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #10

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #11

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #11

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #12

attributes #0 = { nounwind "amdgpu-cluster-dims"="1,1,1" "amdgpu-flat-work-group-size"="1,512" "denormal-fp-math-f32"="ieee" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { nounwind }
attributes #7 = { convergent nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nofree nosync nounwind willreturn }
attributes #9 = { nocallback nofree nounwind willreturn }
attributes #10 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #11 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #12 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2, !3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!2 = !{!"Triton version 3.9.0+gitd39a3959"}
!3 = !{!"LLVM version 24.0.0git (ce3529423abda3fc4ad0b542daa50af21e539a29)"}
!4 = !{!5}
!5 = !{!"amdg.AsyncCopies", !6, !"Scope containing all AsyncCopyGlobalToLocal and BufferLoadToLocal ops"}
!6 = !{!"amdg.AsyncOps", !"Domain to hold alias scopes to specify aliasing information between AsyncCopyGlobalToLocal, BufferLoadToLocal and LocalLoad ops"}
!7 = !{!"amdgpu-synchronize-as", !"local"}
