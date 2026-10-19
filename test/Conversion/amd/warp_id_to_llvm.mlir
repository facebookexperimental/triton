// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942  | FileCheck %s --check-prefixes=CHECK,GFX9
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950  | FileCheck %s --check-prefixes=CHECK,GFX9
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 | FileCheck %s --check-prefixes=CHECK,GFX12
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s --check-prefixes=CHECK,GFX12
// RUN: triton-opt %s -split-input-file --tritonamdgpu-automatic-wave-specialization | FileCheck %s --check-prefix=WS
// RUN: triton-opt %s -split-input-file --tritonamdgpu-automatic-wave-specialization --tritonamdgpu-automatic-wave-specialization | FileCheck %s --check-prefix=WS
// RUN: triton-opt %s -split-input-file --tritonamdgpu-automatic-wave-specialization --canonicalize-llvm-ir --sccp --canonicalize | FileCheck %s --check-prefix=CLEAN
// RUN: triton-opt %s -split-input-file --tritonamdgpu-automatic-wave-specialization=max-extra-ops=0 | FileCheck %s --check-prefix=LIMIT
// RUN: triton-opt %s -split-input-file --tritonamdgpu-automatic-wave-specialization=max-extra-ops=0 --canonicalize-llvm-ir --sccp --canonicalize | FileCheck %s --check-prefix=RANGE

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: @wave_id
tt.func public @wave_id() {
  //       GFX9: %[[C64:.+]] = llvm.mlir.constant(64 : i32) : i32
  //  GFX9-NEXT: %[[IDX:.+]] = rocdl.workitem.id.x : i32
  //  GFX9-NEXT: %[[C63:.+]] = llvm.mlir.constant(63 : i32) : i32
  //  GFX9-NEXT: %[[AND:.+]] = llvm.and %[[IDX]], %[[C63]] : i32
  //  GFX9-NEXT: %[[DIV:.+]] = llvm.udiv %[[AND]], %[[C64]] : i32
  //  GFX9-NEXT: %{{.+}} = rocdl.readfirstlane %[[DIV]] : i32

  // GFX12-NEXT: rocdl.wave.id
  //      CHECK: scf.for

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %1 = "ttg.warp_id"() : () -> i32
    scf.yield
  }
  tt.return
}

}

// -----
module attributes {"ttg.threads-per-warp" = 32 : i32} {
  // WS-LABEL: llvm.func @composed_wave32
  // WS-SAME: tritonamdgpu.wave_specialized
  // WS: llvm.udiv
  // WS: llvm.lshr
  // WS: llvm.cond_br
  // CLEAN-LABEL: llvm.func @composed_wave32
  // CLEAN-SAME: tritonamdgpu.wave_specialized
  // CLEAN-NOT: llvm.select
  // CLEAN: llvm.return
  llvm.func @composed_wave32(%out: !llvm.ptr) {
    %tid = rocdl.workitem.id.x : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %part = llvm.udiv %tid, %c4 : i32
    %wave = llvm.lshr %part, %c3 : i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    %p = llvm.icmp "eq" %wave, %zero : i32
    %v = llvm.select %p, %wave, %zero : i1, i32
    llvm.store %v, %out : i32, !llvm.ptr
    llvm.return
  }
}

// -----
module {
  // WS-LABEL: llvm.func @missing_wave_size
  // WS-NOT: tritonamdgpu.wave_specialized
  // WS: llvm.return
  llvm.func @missing_wave_size(%out: !llvm.ptr) {
    %tid = rocdl.workitem.id.x : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %wave = llvm.udiv %tid, %c64 : i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    %p = llvm.icmp "eq" %wave, %zero : i32
    %v = llvm.select %p, %wave, %zero : i1, i32
    llvm.store %v, %out : i32, !llvm.ptr
    llvm.return
  }
}

// -----
// The ordinary conversion runs above leave these already-lowered functions alone.
module attributes {"ttg.threads-per-warp" = 64 : i32} {
  // WS-LABEL: llvm.func @specialize_loop
  // WS-SAME: tritonamdgpu.wave_specialized
  // WS: %[[TID:.*]] = rocdl.workitem.id.x
  // WS: %[[P:.*]] = llvm.icmp "eq"
  // WS-NEXT: %[[EXT:.*]] = llvm.zext %[[P]]
  // WS-NEXT: %[[FREEZE:.*]] = llvm.freeze %[[EXT]]
  // WS-NEXT: %[[BROADCAST:.*]] = rocdl.readfirstlane %[[FREEZE]]
  // WS-NEXT: %[[DISPATCH:.*]] = llvm.trunc %[[BROADCAST]]
  // WS-NEXT: llvm.cond_br %[[DISPATCH]], ^[[TRUE:bb[0-9]+]], ^[[FALSE:bb[0-9]+]]
  // WS: ^[[TRUE]]:
  // WS-NEXT: %[[T:.*]] = llvm.mlir.constant(true)
  // WS: llvm.store
  // WS: llvm.br ^[[LOOP_T:bb[0-9]+]](
  // WS: ^[[LOOP_T]](%[[I_T:.*]]: i32):
  // WS: llvm.select %[[T]]
  // WS: llvm.cond_br %[[T]]
  // WS: llvm.cond_br %{{.*}}, ^[[LOOP_T]](
  // WS: ^[[FALSE]]:
  // WS-NEXT: %[[F:.*]] = llvm.mlir.constant(false)
  // WS: llvm.store
  // WS: llvm.br ^[[LOOP_F:bb[0-9]+]](
  // WS: ^[[LOOP_F]](%[[I_F:.*]]: i32):
  // WS: llvm.select %[[F]]
  // WS: llvm.cond_br %[[F]]
  // WS: llvm.cond_br %{{.*}}, ^[[LOOP_F]](
  // WS: llvm.return
  // CLEAN-LABEL: llvm.func @specialize_loop
  // CLEAN-SAME: tritonamdgpu.wave_specialized
  // CLEAN: llvm.cond_br
  // CLEAN-NOT: llvm.select
  // CLEAN-COUNT-4: rocdl.s.barrier
  // CLEAN-NOT: rocdl.s.barrier
  // CLEAN: llvm.return
  // LIMIT-LABEL: llvm.func @specialize_loop
  // LIMIT-NOT: tritonamdgpu.wave_specialized
  // LIMIT: llvm.select
  // LIMIT-NOT: tritonamdgpu.wave_specialized
  llvm.func @specialize_loop(%out: !llvm.ptr, %shared: !llvm.ptr<3>, %n: i32) {
    %tid = rocdl.workitem.id.x : i32
    %size = llvm.mlir.constant(64 : i32) : i32
    %wave = llvm.udiv %tid, %size : i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    // Retained work stays in source order, even before the predicate definition.
    %one = llvm.mlir.constant(1 : i32) : i32
    %unrelated = llvm.add %n, %one : i32
    llvm.store %unrelated, %out : i32, !llvm.ptr
    %p = llvm.icmp "eq" %wave, %zero : i32
    llvm.br ^loop(%zero : i32)
  ^loop(%i: i32):
    %value = llvm.select %p, %i, %n : i1, i32
    llvm.store %value, %shared : i32, !llvm.ptr<3>
    llvm.cond_br %p, ^a, ^b
  ^a:
    rocdl.s.barrier
    llvm.br ^join
  ^b:
    rocdl.s.barrier
    llvm.br ^join
  ^join:
    %read = llvm.load %shared : !llvm.ptr<3> -> i32
    llvm.store %read, %out : i32, !llvm.ptr
    rocdl.s.barrier
    %next = llvm.add %i, %one : i32
    %more = llvm.icmp "ult" %next, %n : i32
    llvm.cond_br %more, ^loop(%next : i32), ^exit
  ^exit:
    llvm.return
  }

  // Wave size comes from the IR; tid / 32 is not uniform on a wave64 target.
  // WS-LABEL: llvm.func @wave32
  // WS-NOT: tritonamdgpu.wave_specialized
  // WS: llvm.return
  llvm.func @wave32(%out: !llvm.ptr) {
    %tid = rocdl.workitem.id.x : i32
    %size = llvm.mlir.constant(32 : i32) : i32
    %wave = llvm.udiv %tid, %size : i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    %p = llvm.icmp "eq" %wave, %zero : i32
    %v = llvm.select %p, %wave, %zero : i1, i32
    llvm.store %v, %out : i32, !llvm.ptr
    llvm.return
  }

  // WS-LABEL: llvm.func @lane_varying
  // WS-NOT: tritonamdgpu.wave_specialized
  // WS: llvm.return
  llvm.func @lane_varying(%out: !llvm.ptr) {
    %tid = rocdl.workitem.id.x : i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    %p = llvm.icmp "eq" %tid, %zero : i32
    %v = llvm.select %p, %tid, %zero : i1, i32
    llvm.store %v, %out : i32, !llvm.ptr
    llvm.return
  }

  // The uniform broadcast cannot be extracted across the retained store.
  // WS-LABEL: llvm.func @blocked_extraction
  // WS-NOT: tritonamdgpu.wave_specialized
  // WS: llvm.return
  llvm.func @blocked_extraction(%ptr: !llvm.ptr, %n: i32) {
    llvm.store %n, %ptr : i32, !llvm.ptr
    %load = llvm.load %ptr : !llvm.ptr -> i32
    %uniform = rocdl.readfirstlane %load : i32
    %p = llvm.icmp "eq" %uniform, %n : i32
    %v = llvm.select %p, %uniform, %n : i1, i32
    llvm.store %v, %ptr : i32, !llvm.ptr
    llvm.return
  }

  // Uniform loop-local predicates must not cause a dispatch inside the loop.
  // WS-LABEL: llvm.func @loop_local
  // WS-NOT: tritonamdgpu.wave_specialized
  // WS: llvm.return
  llvm.func @loop_local(%out: !llvm.ptr, %n: i32) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^loop(%zero : i32)
  ^loop(%i: i32):
    %p = llvm.icmp "ult" %i, %n : i32
    %v = llvm.select %p, %i, %n : i1, i32
    llvm.store %v, %out : i32, !llvm.ptr
    %next = llvm.add %i, %one : i32
    llvm.cond_br %p, ^loop(%next : i32), ^exit
  ^exit:
    llvm.return
  }

  // Device helper arguments do not have the uniform kernel ABI.
  // WS-LABEL: llvm.func internal @helper
  // WS-NOT: tritonamdgpu.wave_specialized
  // WS: llvm.return
  llvm.func internal @helper(%p: i1, %a: i32, %b: i32, %out: !llvm.ptr) {
    %v = llvm.select %p, %a, %b : i1, i32
    llvm.store %v, %out : i32, !llvm.ptr
    llvm.return
  }

  // A kernel-argument predicate is also eligible. Boolean masks and the
  // complemented condition fold, preserving successor operands at both exits.
  // CLEAN-LABEL: llvm.func @boolean_masks
  // CLEAN-SAME: tritonamdgpu.wave_specialized
  // CLEAN-NOT: llvm.and
  // CLEAN-NOT: llvm.or
  // CLEAN-NOT: llvm.xor
  // CLEAN-NOT: llvm.select
  // CLEAN: llvm.return
  llvm.func @boolean_masks(%p: i1, %out: !llvm.ptr) {
    %t = llvm.mlir.constant(true) : i1
    %f = llvm.mlir.constant(false) : i1
    %both = llvm.and %p, %t : i1
    %either = llvm.or %both, %f : i1
    %negated = llvm.xor %either, %t : i1
    %one = llvm.mlir.constant(1 : i32) : i32
    %two = llvm.mlir.constant(2 : i32) : i32
    %selected = llvm.select %negated, %one, %two : i1, i32
    llvm.cond_br %p, ^yes(%selected : i32), ^no(%selected : i32)
  ^yes(%a: i32):
    llvm.store %a, %out : i32, !llvm.ptr
    llvm.return
  ^no(%b: i32):
    llvm.store %b, %out : i32, !llvm.ptr
    llvm.return
  }

  // Fused TDM selects a descriptor pair using ((3 >> wave_id) & 1) != 0.
  // Its selectors can be emitted in several blocks inside the persistent loop.
  // Only one slice is extracted; every equivalent selector is specialized.
  // CLEAN-LABEL: llvm.func @fused_tdm_selectors
  // CLEAN-SAME: tritonamdgpu.wave_specialized
  // CLEAN: rocdl.wave.id
  // CLEAN: llvm.icmp "{{eq|ne}}"
  // CLEAN: llvm.freeze
  // CLEAN: llvm.cond_br
  // CLEAN-NOT: llvm.select
  // CLEAN: llvm.return
  llvm.func @fused_tdm_selectors(%out: !llvm.ptr, %n: i32,
                                 %a: vector<4xi32>, %b: vector<4xi32>) {
    %wave = rocdl.wave.id : i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %mask = llvm.mlir.constant(3 : i32) : i32
    llvm.br ^loop(%zero : i32)
  // Deliberately put a dominated block before its dominator in textual order.
  ^later:
    %shift2 = llvm.lshr %mask, %wave : i32
    %bit2 = llvm.and %shift2, %one : i32
    %p2 = llvm.icmp "ne" %bit2, %zero : i32
    %v2 = llvm.select %p2, %a, %b : i1, vector<4xi32>
    llvm.store %v2, %out : vector<4xi32>, !llvm.ptr
    %next = llvm.add %i, %one : i32
    %more = llvm.icmp "slt" %next, %n : i32
    llvm.cond_br %more, ^loop(%next : i32), ^exit
  ^loop(%i: i32):
    rocdl.s.barrier
    %shift = llvm.lshr %mask, %wave : i32
    %bit = llvm.and %shift, %one : i32
    %p = llvm.icmp "ne" %bit, %zero : i32
    %v = llvm.select %p, %a, %b : i1, vector<4xi32>
    llvm.store %v, %out : vector<4xi32>, !llvm.ptr
    llvm.br ^later
  ^exit:
    llvm.return
  }
}

// -----
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Range propagation through division and casts proves wave in [0,3]. All
  // three predicates describe the same partition, including the complement
  // used to construct scalar descriptor bits. Keep an original loop/barrier.
  // CLEAN-LABEL: llvm.func @range_equivalent_descriptors
  // CLEAN-SAME: tritonamdgpu.wave_specialized
  // CLEAN: rocdl.workitem.id.x range <i32, 0, 128>
  // CLEAN: llvm.icmp "eq"
  // CLEAN: llvm.freeze
  // CLEAN: llvm.cond_br
  // CLEAN-NOT: llvm.select
  // CLEAN-NOT: llvm.xor
  // CLEAN-NOT: llvm.zext
  // CLEAN-NOT: llvm.lshr
  // CLEAN: rocdl.s.barrier
  // CLEAN-NOT: llvm.select
  // CLEAN-NOT: llvm.xor
  // CLEAN-NOT: llvm.zext
  // CLEAN-NOT: llvm.lshr
  // CLEAN: rocdl.s.barrier
  // CLEAN-NOT: llvm.select
  // CLEAN-NOT: llvm.xor
  // CLEAN-NOT: llvm.zext
  // CLEAN-NOT: llvm.lshr
  // CLEAN: llvm.return
  // RANGE-LABEL: llvm.func @range_equivalent_descriptors
  // RANGE-NOT: llvm.lshr
  // RANGE: llvm.icmp "eq"
  // RANGE-NOT: llvm.lshr
  // RANGE: llvm.return
  llvm.func @range_equivalent_descriptors(%out: !llvm.ptr, %more: i1,
                                          %a: vector<4xi32>, %b: vector<4xi32>) {
    %tid = rocdl.workitem.id.x : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %w32 = llvm.udiv %tid, %c32 : i32
    %w64 = llvm.zext %w32 : i32 to i64
    %w = llvm.trunc %w64 : i64 to i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %two = llvm.mlir.constant(2 : i32) : i32
    %three = llvm.mlir.constant(3 : i32) : i32
    llvm.br ^loop
  ^loop:
    rocdl.s.barrier
    %shift = llvm.lshr %three, %w : i32
    %bit = llvm.and %shift, %one : i32
    %p = llvm.icmp "ne" %bit, %zero : i32
    %desc = llvm.select %p, %a, %b : i1, vector<4xi32>
    llvm.store %desc, %out : vector<4xi32>, !llvm.ptr
    %masked = llvm.and %w, %two : i32
    %pa = llvm.icmp "eq" %masked, %zero : i32
    %abit = llvm.zext %pa : i1 to i32
    llvm.store %abit, %out : i32, !llvm.ptr
    %flipped = llvm.xor %w, %two : i32
    %masked_b = llvm.and %flipped, %two : i32
    %pb = llvm.icmp "eq" %masked_b, %zero : i32
    %bbit = llvm.zext %pb : i1 to i32
    llvm.store %bbit, %out : i32, !llvm.ptr
    llvm.cond_br %more, ^loop, ^exit
  ^exit:
    llvm.return
  }

  // Comparisons constant over the workgroup should fold without cloning.
  // RANGE-LABEL: llvm.func @range_constant_comparisons
  // RANGE-NOT: tritonamdgpu.wave_specialized
  // RANGE: %[[TRUE:.*]] = llvm.mlir.constant(true)
  // RANGE-NOT: llvm.icmp
  // RANGE: llvm.store %[[TRUE]]
  // RANGE-NOT: llvm.icmp
  // RANGE: llvm.return
  llvm.func @range_constant_comparisons(%out: !llvm.ptr) {
    %wave = rocdl.wave.id : i32
    %four = llvm.mlir.constant(4 : i32) : i32
    %p = llvm.icmp "ult" %wave, %four : i32
    llvm.store %p, %out : i1, !llvm.ptr
    llvm.return
  }
}

// -----
// Extra partitions must be included in the physical wave bound. With eight
// waves, the lookup and the wave-bit test are no longer equivalent.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32,
                   "ttg.threads-per-warp" = 32 : i32} {
  // RANGE-LABEL: llvm.func @range_extra_waves
  // RANGE: rocdl.wave.id range <i32, 0, 8>
  // RANGE: llvm.lshr
  // RANGE: llvm.return
  llvm.func @range_extra_waves(%out: !llvm.ptr) {
    %wave = rocdl.wave.id : i32
    %three = llvm.mlir.constant(3 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    %shift = llvm.lshr %three, %wave : i32
    %bit = llvm.and %shift, %one : i32
    %p = llvm.icmp "ne" %bit, %zero : i32
    llvm.store %p, %out : i1, !llvm.ptr
    llvm.return
  }
}

// -----
module {
  // Explicit ID ranges work without Triton metadata. Standard CFG propagation
  // joins the incoming values; arithmetic and casts preserve the useful bound.
  // RANGE-LABEL: llvm.func @range_cfg_join
  // RANGE: %[[TRUE:.*]] = llvm.mlir.constant(true)
  // RANGE-NOT: llvm.icmp
  // RANGE: llvm.store %[[TRUE]]
  // RANGE: llvm.return
  llvm.func @range_cfg_join(%out: !llvm.ptr, %choose: i1) {
    %tid = rocdl.workitem.id.x range <i32, 0, 128> : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    // LLVM constants can use an attribute wider than the result type.
    %one = llvm.mlir.constant(1 : i64) : i32
    %three = llvm.mlir.constant(3 : i32) : i32
    %eight = llvm.mlir.constant(8 : i32) : i32
    %w = llvm.udiv %tid, %c32 : i32
    %next = llvm.add %w, %one : i32
    llvm.cond_br %choose, ^join(%next : i32), ^join(%three : i32)
  ^join(%v: i32):
    %wide = llvm.sext %v : i32 to i64
    %narrow = llvm.trunc %wide : i64 to i32
    %p = llvm.icmp "ult" %narrow, %eight : i32
    llvm.store %p, %out : i1, !llvm.ptr
    llvm.return
  }

  // The signed range of this wrapping LLVM range is [-2,1]. Treating it as an
  // ordinary unsigned interval would incorrectly prove the unsigned test.
  // RANGE-LABEL: llvm.func @range_wrapped
  // RANGE: llvm.icmp "ult"
  // RANGE-NOT: llvm.icmp "slt"
  // RANGE: llvm.return
  llvm.func @range_wrapped(%out: !llvm.ptr) {
    %tid = rocdl.workitem.id.x range <i32, -2, 2> : i32
    %two = llvm.mlir.constant(2 : i32) : i32
    %p = llvm.icmp "ult" %tid, %two : i32
    %q = llvm.icmp "slt" %tid, %two : i32
    llvm.store %p, %out : i1, !llvm.ptr
    llvm.store %q, %out : i1, !llvm.ptr
    llvm.return
  }

  // No launch bound is available, and freeze is not a range-preserving op:
  // a poisoned input may become any value. Neither comparison may fold.
  // RANGE-LABEL: llvm.func @range_unknown_and_freeze
  // RANGE: llvm.freeze
  // RANGE: llvm.icmp "ult"
  // RANGE: llvm.icmp "ult"
  // RANGE: llvm.return
  llvm.func @range_unknown_and_freeze(%out: !llvm.ptr) {
    %tid = rocdl.workitem.id.x : i32
    %bounded = rocdl.wave.id range <i32, 0, 4> : i32
    %frozen = llvm.freeze %bounded : i32
    %four = llvm.mlir.constant(4 : i32) : i32
    %p = llvm.icmp "ult" %tid, %four : i32
    %q = llvm.icmp "ult" %frozen, %four : i32
    llvm.store %p, %out : i1, !llvm.ptr
    llvm.store %q, %out : i1, !llvm.ptr
    llvm.return
  }

  // Pointer comparisons return i1 but do not have integer operand ranges.
  // RANGE-LABEL: llvm.func @range_pointer_comparison
  // RANGE: llvm.icmp "eq"
  // RANGE: llvm.store
  // RANGE: llvm.return
  llvm.func @range_pointer_comparison(%out: !llvm.ptr, %other: !llvm.ptr) {
    %p = llvm.icmp "eq" %out, %other : !llvm.ptr
    llvm.store %p, %out : i1, !llvm.ptr
    llvm.return
  }
}
