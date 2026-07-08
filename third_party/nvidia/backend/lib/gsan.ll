; ModuleID = '/data/users/xzhao9/fbtriton/python/triton/experimental/gsan/src/GSanLibrary.cu'
source_filename = "/data/users/xzhao9/fbtriton/python/triton/experimental/gsan/src/GSanLibrary.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%"struct.gsan::Location" = type { ptr, i32 }

@.str = private unnamed_addr constant [10 x i8] c"<unknown>\00", align 1
@.str1 = private unnamed_addr constant [31 x i8] c"Read after write race detected\00", align 1
@.str2 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str3 = private unnamed_addr constant [24 x i8] c"Vector clock overflowed\00", align 1
@.str4 = private unnamed_addr constant [31 x i8] c"Write after read race detected\00", align 1
@.str5 = private unnamed_addr constant [32 x i8] c"Write after write race detected\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn denormal_fpenv(float: preservesign) memory(argmem: read)
define dso_local noundef nonnull ptr @_ZN4gsan13getSourceFileENS_8LocationE(ptr noundef readonly byval(%"struct.gsan::Location") align 8 captures(none) %loc) local_unnamed_addr #0 {
entry:
  %0 = load ptr, ptr %loc, align 8, !tbaa !7
  %cmp = icmp eq ptr %0, null
  %cond = select i1 %cmp, ptr @.str, ptr %0
  ret ptr %cond
}

; Function Attrs: convergent mustprogress nounwind denormal_fpenv(float: preservesign)
define dso_local void @__triton_gsan_load_tensor(ptr noundef %globalState, ptr noundef readonly captures(none) %stackPtr, i32 noundef %numElems, i32 noundef %bytesPerElem, ptr noundef %file, i32 noundef %line) local_unnamed_addr #1 {
entry:
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.smid()
  %1 = ptrtoint ptr %globalState to i64
  %ptr.biased.i.i = add i64 %1, 39
  %cond.i.i = and i64 %ptr.biased.i.i, -8
  %2 = getelementptr i8, ptr %globalState, i64 24
  %globals.val.i = load i16, ptr %2, align 8, !tbaa !11
  %3 = getelementptr i8, ptr %globalState, i64 26
  %globals.val24.i = load i16, ptr %3, align 2, !tbaa !15
  %conv.i.i = zext i16 %globals.val24.i to i64
  %add.i.i = add nuw nsw i64 %conv.i.i, 1
  %conv1.i.i = zext i16 %globals.val.i to i64
  %mul.i.i = shl nuw nsw i64 %conv1.i.i, 1
  %mul3.i.i = mul nuw nsw i64 %mul.i.i, %add.i.i
  %add4.i.i = add nuw nsw i64 %mul3.i.i, 32
  %conv.i = zext i32 %0 to i64
  %mul.i = mul i64 %add4.i.i, %conv.i
  %add3.i = add i64 %mul.i, %cond.i.i
  %4 = inttoptr i64 %add3.i to ptr
  %5 = load ptr, ptr %4, align 8, !tbaa !16
  %cmp.i = icmp eq ptr %5, null
  br i1 %cmp.i, label %if.then.i, label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit

if.then.i:                                        ; preds = %entry
  %6 = load i64, ptr %globalState, align 8, !tbaa !18
  %reserveBase5.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 %6, ptr %reserveBase5.i, align 8, !tbaa !19
  %numReads.i = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i32 0, ptr %numReads.i, align 8, !tbaa !3
  %clockBufferDirty.i = getelementptr inbounds nuw i8, ptr %4, i64 20
  store i32 0, ptr %clockBufferDirty.i, align 4
  %globalsBase1.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 8
  %7 = load i64, ptr %globalsBase1.i.i, align 8, !tbaa !20
  %sub.i.i = sub i64 %1, %7
  %div6.i.i = lshr i64 %sub.i.i, 30
  %numSms.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 20
  %8 = load i16, ptr %numSms.i.i, align 4, !tbaa !21
  %conv.i25.i = zext i16 %8 to i64
  %mul.i26.i = mul nuw nsw i64 %div6.i.i, %conv.i25.i
  %add.i27.i = add nuw nsw i64 %mul.i26.i, %conv.i
  %conv3.i.i = trunc i64 %add.i27.i to i16
  %threadId.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  store i16 %conv3.i.i, ptr %threadId.i, align 4, !tbaa !22
  fence release
  store ptr %globalState, ptr %4, align 8, !tbaa !16
  br label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit

_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit: ; preds = %entry, %if.then.i
  %conv.i4 = sext i32 %numElems to i64
  %mul.i5 = shl nsw i64 %conv.i4, 3
  %add.ptr.i = getelementptr inbounds nuw i8, ptr %stackPtr, i64 %mul.i5
  %cmp9.i = icmp sgt i32 %numElems, 0
  br i1 %cmp9.i, label %for.body.lr.ph.i, label %_ZN4gsan12_GLOBAL__N_110tensorLoadEPNS_11ThreadStateEPKciiNS_8LocationE.exit

for.body.lr.ph.i:                                 ; preds = %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit
  %conv.i.i6 = sext i32 %bytesPerElem to i64
  %reserveBase1.i.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  %lock.i.i = getelementptr inbounds nuw i8, ptr %4, i64 24
  %vectorClock.i.i.i = getelementptr inbounds nuw i8, ptr %4, i64 30
  %threadId10.i.i.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  %numReads40.i.i.i = getelementptr inbounds nuw i8, ptr %4, i64 16
  %and.i.i25.i.i = and i64 %add3.i, -1073741824
  %9 = inttoptr i64 %and.i.i25.i.i to ptr
  %rngSeed.i.i.i = getelementptr inbounds nuw i8, ptr %9, i64 16
  %cmp.i.i.i.i = icmp eq ptr %file, null
  %cond.i.i.i.i = select i1 %cmp.i.i.i.i, ptr @.str, ptr %file
  br label %for.body.i

for.body.i:                                       ; preds = %if.end.i, %for.body.lr.ph.i
  %i.010.i = phi i32 [ 0, %for.body.lr.ph.i ], [ %inc.i, %if.end.i ]
  %idxprom.i = zext nneg i32 %i.010.i to i64
  %arrayidx2.i = getelementptr inbounds nuw i8, ptr %add.ptr.i, i64 %idxprom.i
  %10 = load i8, ptr %arrayidx2.i, align 1, !tbaa !23
  %tobool.not.i = icmp eq i8 %10, 0
  br i1 %tobool.not.i, label %if.end.i, label %if.then.i7

if.then.i7:                                       ; preds = %for.body.i
  %arrayidx.i = getelementptr inbounds nuw [8 x i8], ptr %stackPtr, i64 %idxprom.i
  %11 = load i64, ptr %arrayidx.i, align 8, !tbaa !19
  %add.i.i8 = add i64 %11, %conv.i.i6
  %sub.i.i.i = and i64 %11, -4
  %rem3.i.i.i = and i64 %add.i.i8, 3
  %cmp.i.i.i = icmp eq i64 %rem3.i.i.i, 0
  %sub5.i.i.i = sub nuw nsw i64 4, %rem3.i.i.i
  %cond.i.i.i = select i1 %cmp.i.i.i, i64 0, i64 %sub5.i.i.i
  %add.i.i.i = add i64 %cond.i.i.i, %add.i.i8
  %12 = load i64, ptr %reserveBase1.i.i, align 8, !tbaa !19
  %13 = atomicrmw add ptr %lock.i.i, i32 1 syncscope("block") acquire, align 4
  %cmp.i19.i.i = icmp sgt i32 %13, -1
  br i1 %cmp.i19.i.i, label %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i, label %do.body.i.i.i

do.body.i.i.i:                                    ; preds = %if.then.i7, %do.body.i.i.i
  %14 = load atomic i32, ptr %lock.i.i syncscope("block") acquire, align 8
  %cmp3.not.i.i.i = icmp sgt i32 %14, -1
  br i1 %cmp3.not.i.i.i, label %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i, label %do.body.i.i.i, !llvm.loop !24

_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i: ; preds = %do.body.i.i.i, %if.then.i7
  %cmp27.i.i = icmp ult i64 %sub.i.i.i, %add.i.i.i
  br i1 %cmp27.i.i, label %for.body.i.i.preheader, label %_ZN4gsan12_GLOBAL__N_19readRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i

for.body.i.i.preheader:                           ; preds = %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i
  %invariant.op = sub i64 -549755813888, %12
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i.preheader, %for.inc.i.i
  %addr.028.i.i = phi i64 [ %add8.i.i, %for.inc.i.i ], [ %sub.i.i.i, %for.body.i.i.preheader ]
  %and.i.i.i.i = and i64 %addr.028.i.i, -1099511627776
  %cmp.i20.i.i = icmp eq i64 %and.i.i.i.i, %12
  br i1 %cmp.i20.i.i, label %if.end.i.i, label %for.inc.i.i

if.end.i.i:                                       ; preds = %for.body.i.i
  %sub.i22.reass.i.reass.i.reass.reass = add i64 %addr.028.i.i, %invariant.op
  %div4.i.i.i = lshr exact i64 %sub.i22.reass.i.reass.i.reass.reass, 2
  %mul.i.i.i = mul i64 %div4.i.i.i, 24
  %add.i23.i.i = add i64 %mul.i.i.i, %12
  %15 = inttoptr i64 %add.i23.i.i to ptr
  %lock.i.i.i = getelementptr inbounds nuw i8, ptr %15, i64 22
  br label %while.cond.i.i.i

while.cond.i.i.i:                                 ; preds = %while.cond.i.i.i, %if.end.i.i
  %16 = cmpxchg weak ptr %lock.i.i.i, i16 0, i16 1 acquire monotonic, align 2
  %17 = extractvalue { i16, i1 } %16, 1
  br i1 %17, label %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i, label %while.cond.i.i.i, !llvm.loop !26

_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i: ; preds = %while.cond.i.i.i
  %numReads1.i.i.i = getelementptr inbounds nuw i8, ptr %15, i64 20
  %18 = load i16, ptr %numReads1.i.i.i, align 4, !tbaa !27
  %conv.i.i.i = zext i16 %18 to i32
  %cmp.not.i.i.i = icmp eq i16 %18, -1
  br i1 %cmp.not.i.i.i, label %if.end.i.i.i, label %if.then.i.i.i

if.then.i.i.i:                                    ; preds = %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i
  %inc.i.i.i = add nuw i16 %18, 1
  store i16 %inc.i.i.i, ptr %numReads1.i.i.i, align 4, !tbaa !27
  br label %if.end.i.i.i

if.end.i.i.i:                                     ; preds = %if.then.i.i.i, %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i
  %writeClock.i.i.i = getelementptr inbounds nuw i8, ptr %15, i64 16
  %write.sroa.0.0.copyload.i.i.i = load i16, ptr %writeClock.i.i.i, align 4, !tbaa !22
  %write.sroa.4.0.writeClock.sroa_idx.i.i.i = getelementptr inbounds nuw i8, ptr %15, i64 18
  %write.sroa.4.0.copyload.i.i.i = load i16, ptr %write.sroa.4.0.writeClock.sroa_idx.i.i.i, align 2, !tbaa !23
  %bf.clear.i.i.i = and i16 %write.sroa.4.0.copyload.i.i.i, 4095
  %idxprom.i.i.i = zext nneg i16 %bf.clear.i.i.i to i64
  %arrayidx.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i, i64 %idxprom.i.i.i
  %19 = load i16, ptr %arrayidx.i.i.i, align 2, !tbaa !22
  %cmp6.not.i.i.i = icmp ult i16 %19, %write.sroa.0.0.copyload.i.i.i
  br i1 %cmp6.not.i.i.i, label %if.then7.i.i.i, label %do.end.i.i.i

if.then7.i.i.i:                                   ; preds = %if.end.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str1, ptr noundef nonnull %cond.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #5
  br label %do.end.i.i.i

do.end.i.i.i:                                     ; preds = %if.then7.i.i.i, %if.end.i.i.i
  %20 = load i16, ptr %threadId10.i.i.i, align 4, !tbaa !22
  %idxprom12.i.i.i = zext i16 %20 to i64
  %arrayidx13.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i, i64 %idxprom12.i.i.i
  %21 = load i16, ptr %arrayidx13.i.i.i, align 2, !tbaa !22
  %bf.value.i.i.i = and i16 %20, 4095
  %readClock.sroa.0.0.copyload.i.i.i = load i16, ptr %15, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx22.sroa_idx.i.i.i = getelementptr inbounds nuw i8, ptr %15, i64 2
  %readClock.sroa.4.0.copyload.i.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx22.sroa_idx.i.i.i, align 2, !tbaa !23
  %bf.clear25.i.i.i = and i16 %readClock.sroa.4.0.copyload.i.i.i, 4095
  %cmp28.i.i.i = icmp ne i16 %bf.clear25.i.i.i, %20
  %cmp31.i.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.i.i.i, 0
  %or.cond.not.i.i.i = select i1 %cmp28.i.i.i, i1 %cmp31.i.i.i, i1 false
  br i1 %or.cond.not.i.i.i, label %for.cond.i.i.i, label %if.then32.i.i.i

for.cond.i.i.i:                                   ; preds = %do.end.i.i.i
  %arrayidx22.1.i.i.i = getelementptr inbounds nuw i8, ptr %15, i64 4
  %readClock.sroa.0.0.copyload.1.i.i.i = load i16, ptr %arrayidx22.1.i.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx22.sroa_idx.1.i.i.i = getelementptr inbounds nuw i8, ptr %15, i64 6
  %readClock.sroa.4.0.copyload.1.i.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx22.sroa_idx.1.i.i.i, align 2, !tbaa !23
  %bf.clear25.1.i.i.i = and i16 %readClock.sroa.4.0.copyload.1.i.i.i, 4095
  %cmp28.1.i.i.i = icmp ne i16 %bf.clear25.1.i.i.i, %20
  %cmp31.1.i.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.1.i.i.i, 0
  %or.cond.not.1.i.i.i = select i1 %cmp28.1.i.i.i, i1 %cmp31.1.i.i.i, i1 false
  br i1 %or.cond.not.1.i.i.i, label %for.cond.1.i.i.i, label %if.then32.i.i.i

for.cond.1.i.i.i:                                 ; preds = %for.cond.i.i.i
  %arrayidx22.2.i.i.i = getelementptr inbounds nuw i8, ptr %15, i64 8
  %readClock.sroa.0.0.copyload.2.i.i.i = load i16, ptr %arrayidx22.2.i.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx22.sroa_idx.2.i.i.i = getelementptr inbounds nuw i8, ptr %15, i64 10
  %readClock.sroa.4.0.copyload.2.i.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx22.sroa_idx.2.i.i.i, align 2, !tbaa !23
  %bf.clear25.2.i.i.i = and i16 %readClock.sroa.4.0.copyload.2.i.i.i, 4095
  %cmp28.2.i.i.i = icmp ne i16 %bf.clear25.2.i.i.i, %20
  %cmp31.2.i.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.2.i.i.i, 0
  %or.cond.not.2.i.i.i = select i1 %cmp28.2.i.i.i, i1 %cmp31.2.i.i.i, i1 false
  br i1 %or.cond.not.2.i.i.i, label %for.cond.2.i.i.i, label %if.then32.i.i.i

for.cond.2.i.i.i:                                 ; preds = %for.cond.1.i.i.i
  %arrayidx22.3.i.i.i = getelementptr inbounds nuw i8, ptr %15, i64 12
  %readClock.sroa.0.0.copyload.3.i.i.i = load i16, ptr %arrayidx22.3.i.i.i, align 4, !tbaa !22
  %readClock.sroa.4.0.arrayidx22.sroa_idx.3.i.i.i = getelementptr inbounds nuw i8, ptr %15, i64 14
  %readClock.sroa.4.0.copyload.3.i.i.i = load i16, ptr %readClock.sroa.4.0.arrayidx22.sroa_idx.3.i.i.i, align 2, !tbaa !23
  %bf.clear25.3.i.i.i = and i16 %readClock.sroa.4.0.copyload.3.i.i.i, 4095
  %cmp28.3.i.i.i = icmp ne i16 %bf.clear25.3.i.i.i, %20
  %cmp31.3.i.i.i = icmp ne i16 %readClock.sroa.0.0.copyload.3.i.i.i, 0
  %or.cond.not.3.i.i.i = select i1 %cmp28.3.i.i.i, i1 %cmp31.3.i.i.i, i1 false
  br i1 %or.cond.not.3.i.i.i, label %for.cond.3.i.i.i, label %if.then32.i.i.i

for.cond.3.i.i.i:                                 ; preds = %for.cond.2.i.i.i
  %22 = atomicrmw add ptr %numReads40.i.i.i, i32 1 syncscope("block") monotonic, align 8
  %23 = load i32, ptr %rngSeed.i.i.i, align 16, !tbaa !31
  %conv43.i.i.i = zext i16 %20 to i32
  %mul.i.i.i.i.i.i = mul i32 %22, -862048943
  %or.i.i.i.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %mul.i.i.i.i.i.i, i32 %mul.i.i.i.i.i.i, i32 15)
  %mul1.i.i.i.i.i.i = mul i32 %or.i.i.i.i.i.i.i, 461845907
  %xor.i.i.i.i.i = xor i32 %mul1.i.i.i.i.i.i, %23
  %or.i.i.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %xor.i.i.i.i.i, i32 %xor.i.i.i.i.i, i32 13)
  %mul.i.i.i.i.i = mul i32 %or.i.i.i.i.i.i, 5
  %add.i.i.i.i.i = add i32 %mul.i.i.i.i.i, -430675100
  %mul.i.i6.i.i.i.i = mul i32 %conv43.i.i.i, -862048943
  %or.i.i.i7.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %mul.i.i6.i.i.i.i, i32 %mul.i.i6.i.i.i.i, i32 15)
  %mul1.i.i8.i.i.i.i = mul i32 %or.i.i.i7.i.i.i.i, 461845907
  %xor.i9.i.i.i.i = xor i32 %add.i.i.i.i.i, %mul1.i.i8.i.i.i.i
  %or.i.i10.i.i.i.i = tail call noundef i32 @llvm.fshl.i32(i32 %xor.i9.i.i.i.i, i32 %xor.i9.i.i.i.i, i32 13)
  %mul.i11.i.i.i.i = mul i32 %or.i.i10.i.i.i.i, 5
  %add.i12.i.i.i.i = add i32 %mul.i11.i.i.i.i, -430675100
  %shr.i.i.i.i.i = lshr i32 %add.i12.i.i.i.i, 16
  %24 = xor i32 %add.i12.i.i.i.i, %shr.i.i.i.i.i
  %xor.i13.i.i.i.i = xor i32 %24, 8
  %mul.i14.i.i.i.i = mul i32 %xor.i13.i.i.i.i, -2048144789
  %shr1.i.i.i.i.i = lshr i32 %mul.i14.i.i.i.i, 13
  %xor2.i.i.i.i.i = xor i32 %shr1.i.i.i.i.i, %mul.i14.i.i.i.i
  %mul3.i.i.i.i.i = mul i32 %xor2.i.i.i.i.i, -1028477387
  %shr4.i.i.i.i.i = lshr i32 %mul3.i.i.i.i.i, 16
  %xor5.i.i.i.i.i = xor i32 %shr4.i.i.i.i.i, %mul3.i.i.i.i.i
  %shr.i.i.i = lshr i32 %xor5.i.i.i.i.i, 8
  %rem.i.i.i = urem i32 %shr.i.i.i, %conv.i.i.i
  %cmp46.not.i.i.i = icmp eq i32 %rem.i.i.i, 0
  br i1 %cmp46.not.i.i.i, label %if.end48.i.i.i, label %_ZN4gsan12_GLOBAL__N_16doReadEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i

if.then32.i.i.i:                                  ; preds = %for.cond.2.i.i.i, %for.cond.1.i.i.i, %for.cond.i.i.i, %do.end.i.i.i
  %arrayidx22.lcssa.i.i.i = phi ptr [ %15, %do.end.i.i.i ], [ %arrayidx22.1.i.i.i, %for.cond.i.i.i ], [ %arrayidx22.2.i.i.i, %for.cond.1.i.i.i ], [ %arrayidx22.3.i.i.i, %for.cond.2.i.i.i ]
  %readClock.sroa.4.0.arrayidx22.sroa_idx.le.i.i.i = getelementptr inbounds nuw i8, ptr %arrayidx22.lcssa.i.i.i, i64 2
  store i16 %21, ptr %arrayidx22.lcssa.i.i.i, align 4, !tbaa !22
  store i16 %bf.value.i.i.i, ptr %readClock.sroa.4.0.arrayidx22.sroa_idx.le.i.i.i, align 2, !tbaa !23
  br label %_ZN4gsan12_GLOBAL__N_16doReadEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i

if.end48.i.i.i:                                   ; preds = %for.cond.3.i.i.i
  %rem49.i.i.i = and i32 %xor5.i.i.i.i.i, 3
  %idxprom51.i.i.i = zext nneg i32 %rem49.i.i.i to i64
  %arrayidx52.i.i.i = getelementptr inbounds nuw [4 x i8], ptr %15, i64 %idxprom51.i.i.i
  %scalarClock.sroa.5.0.arrayidx52.sroa_idx.i.i.i = getelementptr inbounds nuw i8, ptr %arrayidx52.i.i.i, i64 2
  store i16 %21, ptr %arrayidx52.i.i.i, align 4, !tbaa !22
  store i16 %bf.value.i.i.i, ptr %scalarClock.sroa.5.0.arrayidx52.sroa_idx.i.i.i, align 2, !tbaa !23
  br label %_ZN4gsan12_GLOBAL__N_16doReadEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i

_ZN4gsan12_GLOBAL__N_16doReadEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i: ; preds = %if.end48.i.i.i, %if.then32.i.i.i, %for.cond.3.i.i.i
  store atomic i16 0, ptr %lock.i.i.i release, align 2
  br label %for.inc.i.i

for.inc.i.i:                                      ; preds = %_ZN4gsan12_GLOBAL__N_16doReadEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i, %for.body.i.i
  %add8.i.i = add i64 %addr.028.i.i, 4
  %cmp.i.i = icmp ult i64 %add8.i.i, %add.i.i.i
  br i1 %cmp.i.i, label %for.body.i.i, label %_ZN4gsan12_GLOBAL__N_19readRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i, !llvm.loop !32

_ZN4gsan12_GLOBAL__N_19readRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i: ; preds = %for.inc.i.i, %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i
  %25 = atomicrmw sub ptr %lock.i.i, i32 1 syncscope("block") monotonic, align 4
  br label %if.end.i

if.end.i:                                         ; preds = %_ZN4gsan12_GLOBAL__N_19readRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i, %for.body.i
  %inc.i = add nuw nsw i32 %i.010.i, 1
  %exitcond.not.i = icmp eq i32 %inc.i, %numElems
  br i1 %exitcond.not.i, label %_ZN4gsan12_GLOBAL__N_110tensorLoadEPNS_11ThreadStateEPKciiNS_8LocationE.exit, label %for.body.i, !llvm.loop !33

_ZN4gsan12_GLOBAL__N_110tensorLoadEPNS_11ThreadStateEPKciiNS_8LocationE.exit: ; preds = %if.end.i, %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit
  ret void
}

; Function Attrs: convergent mustprogress nounwind denormal_fpenv(float: preservesign)
define dso_local void @__triton_gsan_init(ptr noundef %globalState, ptr noundef %file, i32 noundef %line) local_unnamed_addr #1 {
entry:
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.smid()
  %1 = ptrtoint ptr %globalState to i64
  %ptr.biased.i.i.i = add i64 %1, 39
  %cond.i.i.i = and i64 %ptr.biased.i.i.i, -8
  %2 = getelementptr i8, ptr %globalState, i64 24
  %globals.val.i.i = load i16, ptr %2, align 8, !tbaa !11
  %3 = getelementptr i8, ptr %globalState, i64 26
  %globals.val24.i.i = load i16, ptr %3, align 2, !tbaa !15
  %conv.i.i.i = zext i16 %globals.val24.i.i to i64
  %add.i.i.i = add nuw nsw i64 %conv.i.i.i, 1
  %conv1.i.i.i = zext i16 %globals.val.i.i to i64
  %mul.i.i.i = shl nuw nsw i64 %conv1.i.i.i, 1
  %mul3.i.i.i = mul nuw nsw i64 %mul.i.i.i, %add.i.i.i
  %add4.i.i.i = add nuw nsw i64 %mul3.i.i.i, 32
  %conv.i.i = zext i32 %0 to i64
  %mul.i.i = mul i64 %add4.i.i.i, %conv.i.i
  %add3.i.i = add i64 %mul.i.i, %cond.i.i.i
  %4 = inttoptr i64 %add3.i.i to ptr
  %5 = load ptr, ptr %4, align 8, !tbaa !16
  %cmp.i.i = icmp eq ptr %5, null
  br i1 %cmp.i.i, label %if.then.i.i, label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i

if.then.i.i:                                      ; preds = %entry
  %6 = load i64, ptr %globalState, align 8, !tbaa !18
  %reserveBase5.i.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 %6, ptr %reserveBase5.i.i, align 8, !tbaa !19
  %numReads.i.i = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i32 0, ptr %numReads.i.i, align 8, !tbaa !3
  %clockBufferDirty.i.i = getelementptr inbounds nuw i8, ptr %4, i64 20
  store i32 0, ptr %clockBufferDirty.i.i, align 4
  %globalsBase1.i.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 8
  %7 = load i64, ptr %globalsBase1.i.i.i, align 8, !tbaa !20
  %sub.i.i.i = sub i64 %1, %7
  %div6.i.i.i = lshr i64 %sub.i.i.i, 30
  %numSms.i.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 20
  %8 = load i16, ptr %numSms.i.i.i, align 4, !tbaa !21
  %conv.i25.i.i = zext i16 %8 to i64
  %mul.i26.i.i = mul nuw nsw i64 %div6.i.i.i, %conv.i25.i.i
  %add.i27.i.i = add nuw nsw i64 %mul.i26.i.i, %conv.i.i
  %conv3.i.i.i = trunc i64 %add.i27.i.i to i16
  %threadId.i.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  store i16 %conv3.i.i.i, ptr %threadId.i.i, align 4, !tbaa !22
  fence release
  store ptr %globalState, ptr %4, align 8, !tbaa !16
  br label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i

_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i: ; preds = %if.then.i.i, %entry
  %9 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cmp.i = icmp eq i32 %9, 0
  br i1 %cmp.i, label %if.then.i, label %_ZN4gsan12_GLOBAL__N_110initThreadEPNS_11GlobalStateENS_8LocationE.exit

if.then.i:                                        ; preds = %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i
  %globalsBase1.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 8
  %10 = load i64, ptr %globalsBase1.i.i, align 8, !tbaa !20
  %sub.i.i = sub i64 %1, %10
  %div6.i.i = lshr i64 %sub.i.i, 30
  %numSms.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 20
  %11 = load i16, ptr %numSms.i.i, align 4, !tbaa !21
  %conv.i17.i = zext i16 %11 to i64
  %mul.i18.i = mul nuw nsw i64 %div6.i.i, %conv.i17.i
  %add.i.i = add nuw nsw i64 %mul.i18.i, %conv.i.i
  %vectorClock.i = getelementptr inbounds nuw i8, ptr %4, i64 30
  %idxprom.i = and i64 %add.i.i, 65535
  %arrayidx.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i, i64 %idxprom.i
  %12 = load i16, ptr %arrayidx.i, align 2, !tbaa !22
  %cmp6.not.i = icmp eq i16 %12, -1
  br i1 %cmp6.not.i, label %if.then7.i, label %do.end.i

if.then7.i:                                       ; preds = %if.then.i
  %cmp.i19.i = icmp eq ptr %file, null
  %cond.i.i = select i1 %cmp.i19.i, ptr @.str, ptr %file
  tail call void @__assertfail(ptr noundef nonnull @.str3, ptr noundef nonnull %cond.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #5
  %.pre.i = load i16, ptr %arrayidx.i, align 2, !tbaa !22
  br label %do.end.i

do.end.i:                                         ; preds = %if.then7.i, %if.then.i
  %13 = phi i16 [ %.pre.i, %if.then7.i ], [ %12, %if.then.i ]
  %add.i = add i16 %13, 1
  store i16 %add.i, ptr %arrayidx.i, align 2, !tbaa !22
  br label %_ZN4gsan12_GLOBAL__N_110initThreadEPNS_11GlobalStateENS_8LocationE.exit

_ZN4gsan12_GLOBAL__N_110initThreadEPNS_11GlobalStateENS_8LocationE.exit: ; preds = %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit.i, %do.end.i
  ret void
}

; Function Attrs: convergent mustprogress nounwind denormal_fpenv(float: preservesign)
define dso_local void @__triton_gsan_store_tensor(ptr noundef %globalState, ptr noundef readonly captures(none) %stackPtr, i32 noundef %numElems, i32 noundef %bytesPerElem, ptr noundef %file, i32 noundef %line) local_unnamed_addr #1 {
entry:
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.smid()
  %1 = ptrtoint ptr %globalState to i64
  %ptr.biased.i.i = add i64 %1, 39
  %cond.i.i = and i64 %ptr.biased.i.i, -8
  %2 = getelementptr i8, ptr %globalState, i64 24
  %globals.val.i = load i16, ptr %2, align 8, !tbaa !11
  %3 = getelementptr i8, ptr %globalState, i64 26
  %globals.val24.i = load i16, ptr %3, align 2, !tbaa !15
  %conv.i.i = zext i16 %globals.val24.i to i64
  %add.i.i = add nuw nsw i64 %conv.i.i, 1
  %conv1.i.i = zext i16 %globals.val.i to i64
  %mul.i.i = shl nuw nsw i64 %conv1.i.i, 1
  %mul3.i.i = mul nuw nsw i64 %mul.i.i, %add.i.i
  %add4.i.i = add nuw nsw i64 %mul3.i.i, 32
  %conv.i = zext i32 %0 to i64
  %mul.i = mul i64 %add4.i.i, %conv.i
  %add3.i = add i64 %mul.i, %cond.i.i
  %4 = inttoptr i64 %add3.i to ptr
  %5 = load ptr, ptr %4, align 8, !tbaa !16
  %cmp.i = icmp eq ptr %5, null
  br i1 %cmp.i, label %if.then.i, label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit

if.then.i:                                        ; preds = %entry
  %6 = load i64, ptr %globalState, align 8, !tbaa !18
  %reserveBase5.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 %6, ptr %reserveBase5.i, align 8, !tbaa !19
  %numReads.i = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i32 0, ptr %numReads.i, align 8, !tbaa !3
  %clockBufferDirty.i = getelementptr inbounds nuw i8, ptr %4, i64 20
  store i32 0, ptr %clockBufferDirty.i, align 4
  %globalsBase1.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 8
  %7 = load i64, ptr %globalsBase1.i.i, align 8, !tbaa !20
  %sub.i.i = sub i64 %1, %7
  %div6.i.i = lshr i64 %sub.i.i, 30
  %numSms.i.i = getelementptr inbounds nuw i8, ptr %globalState, i64 20
  %8 = load i16, ptr %numSms.i.i, align 4, !tbaa !21
  %conv.i25.i = zext i16 %8 to i64
  %mul.i26.i = mul nuw nsw i64 %div6.i.i, %conv.i25.i
  %add.i27.i = add nuw nsw i64 %mul.i26.i, %conv.i
  %conv3.i.i = trunc i64 %add.i27.i to i16
  %threadId.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  store i16 %conv3.i.i, ptr %threadId.i, align 4, !tbaa !22
  fence release
  store ptr %globalState, ptr %4, align 8, !tbaa !16
  br label %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit

_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit: ; preds = %entry, %if.then.i
  %conv.i4 = sext i32 %numElems to i64
  %mul.i5 = shl nsw i64 %conv.i4, 3
  %add.ptr.i = getelementptr inbounds nuw i8, ptr %stackPtr, i64 %mul.i5
  %cmp9.i = icmp sgt i32 %numElems, 0
  br i1 %cmp9.i, label %for.body.lr.ph.i, label %_ZN4gsan12_GLOBAL__N_111tensorStoreEPNS_11ThreadStateEPKciiNS_8LocationE.exit

for.body.lr.ph.i:                                 ; preds = %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit
  %conv.i.i6 = sext i32 %bytesPerElem to i64
  %reserveBase1.i.i = getelementptr inbounds nuw i8, ptr %4, i64 8
  %lock.i.i = getelementptr inbounds nuw i8, ptr %4, i64 24
  %vectorClock.i.i.i = getelementptr inbounds nuw i8, ptr %4, i64 30
  %threadId22.i.i.i = getelementptr inbounds nuw i8, ptr %4, i64 28
  %cmp.i.i.i.i = icmp eq ptr %file, null
  %cond.i.i.i.i = select i1 %cmp.i.i.i.i, ptr @.str, ptr %file
  br label %for.body.i

for.body.i:                                       ; preds = %if.end.i, %for.body.lr.ph.i
  %i.010.i = phi i32 [ 0, %for.body.lr.ph.i ], [ %inc.i, %if.end.i ]
  %idxprom.i = zext nneg i32 %i.010.i to i64
  %arrayidx2.i = getelementptr inbounds nuw i8, ptr %add.ptr.i, i64 %idxprom.i
  %9 = load i8, ptr %arrayidx2.i, align 1, !tbaa !23
  %tobool.not.i = icmp eq i8 %9, 0
  br i1 %tobool.not.i, label %if.end.i, label %if.then.i7

if.then.i7:                                       ; preds = %for.body.i
  %arrayidx.i = getelementptr inbounds nuw [8 x i8], ptr %stackPtr, i64 %idxprom.i
  %10 = load i64, ptr %arrayidx.i, align 8, !tbaa !19
  %add.i.i8 = add i64 %10, %conv.i.i6
  %sub.i.i.i = and i64 %10, -4
  %rem3.i.i.i = and i64 %add.i.i8, 3
  %cmp.i.i.i = icmp eq i64 %rem3.i.i.i, 0
  %sub5.i.i.i = sub nuw nsw i64 4, %rem3.i.i.i
  %cond.i.i.i = select i1 %cmp.i.i.i, i64 0, i64 %sub5.i.i.i
  %add.i.i.i = add i64 %cond.i.i.i, %add.i.i8
  %11 = load i64, ptr %reserveBase1.i.i, align 8, !tbaa !19
  %12 = atomicrmw add ptr %lock.i.i, i32 1 syncscope("block") acquire, align 4
  %cmp.i19.i.i = icmp sgt i32 %12, -1
  br i1 %cmp.i19.i.i, label %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i, label %do.body.i.i.i

do.body.i.i.i:                                    ; preds = %if.then.i7, %do.body.i.i.i
  %13 = load atomic i32, ptr %lock.i.i syncscope("block") acquire, align 8
  %cmp3.not.i.i.i = icmp sgt i32 %13, -1
  br i1 %cmp3.not.i.i.i, label %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i, label %do.body.i.i.i, !llvm.loop !24

_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i: ; preds = %do.body.i.i.i, %if.then.i7
  %cmp26.i.i = icmp ult i64 %sub.i.i.i, %add.i.i.i
  br i1 %cmp26.i.i, label %for.body.i.i.preheader, label %_ZN4gsan12_GLOBAL__N_110writeRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i

for.body.i.i.preheader:                           ; preds = %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i
  %invariant.op = sub i64 -549755813888, %11
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.body.i.i.preheader, %for.inc.i.i
  %addr.027.i.i = phi i64 [ %add8.i.i, %for.inc.i.i ], [ %sub.i.i.i, %for.body.i.i.preheader ]
  %and.i.i.i.i = and i64 %addr.027.i.i, -1099511627776
  %cmp.i20.i.i = icmp eq i64 %and.i.i.i.i, %11
  br i1 %cmp.i20.i.i, label %if.end.i.i, label %for.inc.i.i

if.end.i.i:                                       ; preds = %for.body.i.i
  %sub.i22.reass.i.reass.i.reass.reass = add i64 %addr.027.i.i, %invariant.op
  %div4.i.i.i = lshr exact i64 %sub.i22.reass.i.reass.i.reass.reass, 2
  %mul.i.i.i = mul i64 %div4.i.i.i, 24
  %add.i23.i.i = add i64 %mul.i.i.i, %11
  %14 = inttoptr i64 %add.i23.i.i to ptr
  %lock.i.i.i = getelementptr inbounds nuw i8, ptr %14, i64 22
  br label %while.cond.i.i.i

while.cond.i.i.i:                                 ; preds = %while.cond.i.i.i, %if.end.i.i
  %15 = cmpxchg weak ptr %lock.i.i.i, i16 0, i16 1 acquire monotonic, align 2
  %16 = extractvalue { i16, i1 } %15, 1
  br i1 %16, label %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i, label %while.cond.i.i.i, !llvm.loop !26

_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i: ; preds = %while.cond.i.i.i
  %read.sroa.0.0.copyload.i.i.i = load i16, ptr %14, align 4, !tbaa !22
  %read.sroa.4.0.arrayidx.sroa_idx.i.i.i = getelementptr inbounds nuw i8, ptr %14, i64 2
  %read.sroa.4.0.copyload.i.i.i = load i16, ptr %read.sroa.4.0.arrayidx.sroa_idx.i.i.i, align 2, !tbaa !23
  %bf.clear.i.i.i = and i16 %read.sroa.4.0.copyload.i.i.i, 4095
  %idxprom1.i.i.i = zext nneg i16 %bf.clear.i.i.i to i64
  %arrayidx2.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i, i64 %idxprom1.i.i.i
  %17 = load i16, ptr %arrayidx2.i.i.i, align 2, !tbaa !22
  %cmp4.not.i.i.i = icmp ult i16 %17, %read.sroa.0.0.copyload.i.i.i
  br i1 %cmp4.not.i.i.i, label %if.then.i.i.i, label %do.end.i.i.i

if.then.i.i.i:                                    ; preds = %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #5
  br label %do.end.i.i.i

do.end.i.i.i:                                     ; preds = %if.then.i.i.i, %_ZN4gsan12_GLOBAL__N_113acquireShadowEm.exit.i.i
  %arrayidx.1.i.i.i = getelementptr inbounds nuw i8, ptr %14, i64 4
  %read.sroa.0.0.copyload.1.i.i.i = load i16, ptr %arrayidx.1.i.i.i, align 4, !tbaa !22
  %read.sroa.4.0.arrayidx.sroa_idx.1.i.i.i = getelementptr inbounds nuw i8, ptr %14, i64 6
  %read.sroa.4.0.copyload.1.i.i.i = load i16, ptr %read.sroa.4.0.arrayidx.sroa_idx.1.i.i.i, align 2, !tbaa !23
  %bf.clear.1.i.i.i = and i16 %read.sroa.4.0.copyload.1.i.i.i, 4095
  %idxprom1.1.i.i.i = zext nneg i16 %bf.clear.1.i.i.i to i64
  %arrayidx2.1.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i, i64 %idxprom1.1.i.i.i
  %18 = load i16, ptr %arrayidx2.1.i.i.i, align 2, !tbaa !22
  %cmp4.not.1.i.i.i = icmp ult i16 %18, %read.sroa.0.0.copyload.1.i.i.i
  br i1 %cmp4.not.1.i.i.i, label %if.then.1.i.i.i, label %do.end.1.i.i.i

if.then.1.i.i.i:                                  ; preds = %do.end.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #5
  br label %do.end.1.i.i.i

do.end.1.i.i.i:                                   ; preds = %if.then.1.i.i.i, %do.end.i.i.i
  %arrayidx.2.i.i.i = getelementptr inbounds nuw i8, ptr %14, i64 8
  %read.sroa.0.0.copyload.2.i.i.i = load i16, ptr %arrayidx.2.i.i.i, align 4, !tbaa !22
  %read.sroa.4.0.arrayidx.sroa_idx.2.i.i.i = getelementptr inbounds nuw i8, ptr %14, i64 10
  %read.sroa.4.0.copyload.2.i.i.i = load i16, ptr %read.sroa.4.0.arrayidx.sroa_idx.2.i.i.i, align 2, !tbaa !23
  %bf.clear.2.i.i.i = and i16 %read.sroa.4.0.copyload.2.i.i.i, 4095
  %idxprom1.2.i.i.i = zext nneg i16 %bf.clear.2.i.i.i to i64
  %arrayidx2.2.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i, i64 %idxprom1.2.i.i.i
  %19 = load i16, ptr %arrayidx2.2.i.i.i, align 2, !tbaa !22
  %cmp4.not.2.i.i.i = icmp ult i16 %19, %read.sroa.0.0.copyload.2.i.i.i
  br i1 %cmp4.not.2.i.i.i, label %if.then.2.i.i.i, label %do.end.2.i.i.i

if.then.2.i.i.i:                                  ; preds = %do.end.1.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #5
  br label %do.end.2.i.i.i

do.end.2.i.i.i:                                   ; preds = %if.then.2.i.i.i, %do.end.1.i.i.i
  %arrayidx.3.i.i.i = getelementptr inbounds nuw i8, ptr %14, i64 12
  %read.sroa.0.0.copyload.3.i.i.i = load i16, ptr %arrayidx.3.i.i.i, align 4, !tbaa !22
  %read.sroa.4.0.arrayidx.sroa_idx.3.i.i.i = getelementptr inbounds nuw i8, ptr %14, i64 14
  %read.sroa.4.0.copyload.3.i.i.i = load i16, ptr %read.sroa.4.0.arrayidx.sroa_idx.3.i.i.i, align 2, !tbaa !23
  %bf.clear.3.i.i.i = and i16 %read.sroa.4.0.copyload.3.i.i.i, 4095
  %idxprom1.3.i.i.i = zext nneg i16 %bf.clear.3.i.i.i to i64
  %arrayidx2.3.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i, i64 %idxprom1.3.i.i.i
  %20 = load i16, ptr %arrayidx2.3.i.i.i, align 2, !tbaa !22
  %cmp4.not.3.i.i.i = icmp ult i16 %20, %read.sroa.0.0.copyload.3.i.i.i
  br i1 %cmp4.not.3.i.i.i, label %if.then.3.i.i.i, label %do.end.3.i.i.i

if.then.3.i.i.i:                                  ; preds = %do.end.2.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str4, ptr noundef nonnull %cond.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #5
  br label %do.end.3.i.i.i

do.end.3.i.i.i:                                   ; preds = %if.then.3.i.i.i, %do.end.2.i.i.i
  %writeClock.i.i.i = getelementptr inbounds nuw i8, ptr %14, i64 16
  %write.sroa.0.0.copyload.i.i.i = load i16, ptr %writeClock.i.i.i, align 4, !tbaa !22
  %write.sroa.4.0.writeClock.sroa_idx.i.i.i = getelementptr inbounds nuw i8, ptr %14, i64 18
  %write.sroa.4.0.copyload.i.i.i = load i16, ptr %write.sroa.4.0.writeClock.sroa_idx.i.i.i, align 2, !tbaa !23
  %bf.clear8.i.i.i = and i16 %write.sroa.4.0.copyload.i.i.i, 4095
  %idxprom9.i.i.i = zext nneg i16 %bf.clear8.i.i.i to i64
  %arrayidx10.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i, i64 %idxprom9.i.i.i
  %21 = load i16, ptr %arrayidx10.i.i.i, align 2, !tbaa !22
  %cmp14.not.i.i.i = icmp ult i16 %21, %write.sroa.0.0.copyload.i.i.i
  br i1 %cmp14.not.i.i.i, label %if.then15.i.i.i, label %_ZN4gsan12_GLOBAL__N_17doWriteEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i

if.then15.i.i.i:                                  ; preds = %do.end.3.i.i.i
  tail call void @__assertfail(ptr noundef nonnull @.str5, ptr noundef nonnull %cond.i.i.i.i, i32 noundef %line, ptr noundef nonnull @.str2, i64 noundef 1) #5
  br label %_ZN4gsan12_GLOBAL__N_17doWriteEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i

_ZN4gsan12_GLOBAL__N_17doWriteEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i: ; preds = %if.then15.i.i.i, %do.end.3.i.i.i
  %22 = load i16, ptr %threadId22.i.i.i, align 4, !tbaa !22
  %idxprom24.i.i.i = zext i16 %22 to i64
  %arrayidx25.i.i.i = getelementptr inbounds nuw [2 x i8], ptr %vectorClock.i.i.i, i64 %idxprom24.i.i.i
  %23 = load i16, ptr %arrayidx25.i.i.i, align 2, !tbaa !22
  %bf.value.i.i.i = and i16 %22, 4095
  store i16 %23, ptr %writeClock.i.i.i, align 4, !tbaa !22
  store i16 %bf.value.i.i.i, ptr %write.sroa.4.0.writeClock.sroa_idx.i.i.i, align 2, !tbaa !23
  store atomic i16 0, ptr %lock.i.i.i release, align 2
  br label %for.inc.i.i

for.inc.i.i:                                      ; preds = %_ZN4gsan12_GLOBAL__N_17doWriteEPNS_11ThreadStateEPNS_10ShadowCellENS_8LocationE.exit.i.i, %for.body.i.i
  %add8.i.i = add i64 %addr.027.i.i, 4
  %cmp.i.i = icmp ult i64 %add8.i.i, %add.i.i.i
  br i1 %cmp.i.i, label %for.body.i.i, label %_ZN4gsan12_GLOBAL__N_110writeRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i, !llvm.loop !34

_ZN4gsan12_GLOBAL__N_110writeRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i: ; preds = %for.inc.i.i, %_ZN4gsan12_GLOBAL__N_117rwLockAcquireReadERj.exit.i.i
  %24 = atomicrmw sub ptr %lock.i.i, i32 1 syncscope("block") monotonic, align 4
  br label %if.end.i

if.end.i:                                         ; preds = %_ZN4gsan12_GLOBAL__N_110writeRangeEPNS_11ThreadStateEmiNS_8LocationE.exit.i, %for.body.i
  %inc.i = add nuw nsw i32 %i.010.i, 1
  %exitcond.not.i = icmp eq i32 %inc.i, %numElems
  br i1 %exitcond.not.i, label %_ZN4gsan12_GLOBAL__N_111tensorStoreEPNS_11ThreadStateEPKciiNS_8LocationE.exit, label %for.body.i, !llvm.loop !35

_ZN4gsan12_GLOBAL__N_111tensorStoreEPNS_11ThreadStateEPKciiNS_8LocationE.exit: ; preds = %if.end.i, %_ZN4gsan12_GLOBAL__N_114getThreadStateEPNS_11GlobalStateE.exit
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.smid() #2

; Function Attrs: convergent nounwind denormal_fpenv(float: preservesign)
declare dso_local void @__assertfail(ptr noundef, ptr noundef, i32 noundef, ptr noundef, i64 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshl.i32(i32, i32, i32) #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn denormal_fpenv(float: preservesign) memory(argmem: read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx91,+sm_80" "uniform-work-group-size" }
attributes #1 = { convergent mustprogress nounwind denormal_fpenv(float: preservesign) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx91,+sm_80" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nounwind denormal_fpenv(float: preservesign) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx91,+sm_80" "uniform-work-group-size" }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!llvm.errno.tbaa = !{!3}

!0 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project 62b7cf9623fc310525f39ed69aaecc318a909731)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !9, i64 0}
!8 = !{!"_ZTSN4gsan8LocationE", !9, i64 0, !4, i64 8}
!9 = !{!"p1 omnipotent char", !10, i64 0}
!10 = !{!"any pointer", !5, i64 0}
!11 = !{!12, !14, i64 24}
!12 = !{!"_ZTSN4gsan11GlobalStateE", !13, i64 0, !13, i64 8, !4, i64 16, !14, i64 20, !14, i64 22, !14, i64 24, !14, i64 26}
!13 = !{!"long", !5, i64 0}
!14 = !{!"short", !5, i64 0}
!15 = !{!12, !14, i64 26}
!16 = !{!17, !17, i64 0}
!17 = !{!"p1 _ZTSN4gsan11GlobalStateE", !10, i64 0}
!18 = !{!12, !13, i64 0}
!19 = !{!13, !13, i64 0}
!20 = !{!12, !13, i64 8}
!21 = !{!12, !14, i64 20}
!22 = !{!14, !14, i64 0}
!23 = !{!5, !5, i64 0}
!24 = distinct !{!24, !25}
!25 = !{!"llvm.loop.mustprogress"}
!26 = distinct !{!26, !25}
!27 = !{!28, !14, i64 20}
!28 = !{!"_ZTSN4gsan10ShadowCellE", !5, i64 0, !29, i64 16, !14, i64 20, !14, i64 22}
!29 = !{!"_ZTSN4gsan11ScalarClockE", !14, i64 0, !14, i64 2, !30, i64 3}
!30 = !{!"_ZTSN4gsan11AtomicScopeE", !5, i64 0}
!31 = !{!12, !4, i64 16}
!32 = distinct !{!32, !25}
!33 = distinct !{!33, !25}
!34 = distinct !{!34, !25}
!35 = distinct !{!35, !25}
