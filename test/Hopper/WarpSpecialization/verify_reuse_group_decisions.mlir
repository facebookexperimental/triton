// REQUIRES: asserts
//
// Directly checks the verifyReuseGroup2 / orderReuseGroupChain DECISIONS (via the
// `nvgpu-ws-utility` debug output, which now names each channel) for the
// reuse-legality cases, so a change to the shared dependency predicate
// (dependsThroughMemory / getRootBuffer) or the N-way chain ordering is caught
// here. Each RUN reuses an existing sibling fixture (no IR duplicated).
//
// ------------------------------------------------------------------------------
// CASE 1 - real 2-way reuse (ACCEPT): dpT and dq share one TMEM slot; dpT's
// consumer feeds dq's producer (dpT -> dsT -> dq), so a dependency chain exists.
// RUN: triton-opt %S/reuse_group_2buffer.mlir --nvgpu-test-ws-code-partition="num-buffers=1" --debug-only=nvgpu-ws-utility 2>&1 | FileCheck %s --check-prefix=ACCEPT2
// ACCEPT2: verifyReuseGroup2: TMEM channels {{[0-9]+}}/{{[0-9]+}} (dpT/dq) overlap=1 chain=1
//
// ------------------------------------------------------------------------------
// CASE 2 - real 2-way reuse through a MULTI-BUFFER subview (ACCEPT): same dpT->dq
// chain but dsT lives in a 1x128x128 alloc and the store/read use separate
// memdesc_index slot-views; only getRootBuffer connects them -> chain=1.
// RUN: triton-opt %S/reuse_group_2buffer_multibuf.mlir --nvgpu-test-ws-code-partition="num-buffers=1" --debug-only=nvgpu-ws-utility 2>&1 | FileCheck %s --check-prefix=MULTIBUF
// MULTIBUF: verifyReuseGroup2: TMEM channels {{[0-9]+}}/{{[0-9]+}} (dpT/dq) overlap=1 chain=1
//
// ------------------------------------------------------------------------------
// CASE 3 - orderable N>=3 reuse (ACCEPT): {dpT,dsT,dq} share one TMEM slot and dk
// reads dsT before dq overwrites it, so orderReuseGroupChain finds a unique order
// and emits the reuse WAR (no fatal).
// RUN: triton-opt %S/ws_code_partition_tmem_3group_chain.mlir --nvgpu-test-ws-code-partition="num-buffers=1" --debug-only=nvgpu-ws-utility 2>&1 | FileCheck %s --check-prefix=ACCEPTN
// ACCEPTN-NOT: no unique dependency-chain order
// ACCEPTN: needExplicitReuseWait: explicit wait needed for earlyChannel {{[0-9]+}} (dpT) and lateChannel {{[0-9]+}} (dq)
//
// ------------------------------------------------------------------------------
// CASE 4 - unorderable N>=3 reuse (REJECT/fast-fail): same {dpT,dsT,dq} slot but
// dq overwrites BEFORE dk reads dsT, so no unique order exists and code
// partitioning must fail fast instead of emitting an unsafe/deadlocking barrier.
// RUN: not --crash triton-opt %S/ws_code_partition_tmem_3group_no_chain.mlir --nvgpu-test-ws-code-partition="num-buffers=1" 2>&1 | FileCheck %s --check-prefix=REJECTN
// REJECTN: TMEM reuse group with >= 3 buffers has no unique dependency-chain order
// REJECTN-SAME: buffer.id=8
// REJECTN-SAME: dpT
// REJECTN-SAME: dsT
// REJECTN-SAME: dq
