/*
 * Deterministic SKC scaffold for expert manual CUDA lowering.
 * This file intentionally contains no executable kernel.
 * Complete and review the manifests before authoring CUDA.
 */
#error "Manual CUDA lowering is required; this scaffold is not executable"

// Twill IR sha256: f07cba4e72bea3ddcba19b84097e406bf3996a0f14a2cf4eeca3de25f1161d69
// Scheduled instructions (schedule facts only):
// {"group": 0, "id": 0, "lanes": [0], "offset": 0, "op_kind": "tt.descriptor_load", "op_ref": "op_1257373504", "stage": 0}
// {"group": 1, "id": 1, "lanes": [0], "offset": 0, "op_kind": "ttg.local_alloc", "op_ref": "op_1257785952", "stage": 0}
// {"group": 0, "id": 2, "lanes": [0], "offset": 0, "op_kind": "tt.descriptor_load", "op_ref": "op_1257061424", "stage": 0}
// {"group": 2, "id": 3, "lanes": [0], "offset": 0, "op_kind": "ttg.local_alloc", "op_ref": "op_1257786096", "stage": 0}
// {"group": 1, "id": 4, "lanes": [0, 1, 2, 3], "offset": 0, "op_kind": "tt.splat", "op_ref": "op_1257775824", "stage": 0}
// {"group": 3, "id": 5, "lanes": [0, 1, 2, 3], "offset": 1, "op_kind": "arith.addi", "op_ref": "op_1257803584", "stage": 0}
// {"group": 4, "id": 6, "lanes": [0, 1, 2, 3], "offset": 1, "op_kind": "tt.addptr", "op_ref": "op_1257801792", "stage": 0}
// {"group": 0, "id": 7, "lanes": [0], "offset": 2, "op_kind": "tt.load", "op_ref": "op_1257788816", "stage": 0}
// {"group": 4, "id": 8, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "tt.addptr", "op_ref": "op_1257803184", "stage": 0}
// {"group": 0, "id": 9, "lanes": [0], "offset": 3, "op_kind": "tt.load", "op_ref": "op_1257792768", "stage": 0}
// {"group": 5, "id": 10, "lanes": [0], "offset": 0, "op_kind": "ttg.memdesc_trans", "op_ref": "op_1257802112", "stage": 0}
// {"group": 2, "id": 11, "lanes": [1], "offset": 1, "op_kind": "ttng.tc_gen5_mma", "op_ref": "op_1257063168", "stage": 0}
// {"group": 6, "id": 12, "lanes": [0, 1, 2, 3], "offset": 4, "op_kind": "ttng.tmem_load", "op_ref": "op_1257785328", "stage": 0}
// {"group": 2, "id": 13, "lanes": [2, 3, 4, 5], "offset": 5, "op_kind": "arith.mulf", "op_ref": "op_1257789488", "stage": 0}
// {"group": 3, "id": 14, "lanes": [0, 1, 2, 3], "offset": 3, "op_kind": "tt.expand_dims", "op_ref": "op_1257789840", "stage": 0}
// {"group": 6, "id": 15, "lanes": [0, 4, 5, 6], "offset": 4, "op_kind": "tt.broadcast", "op_ref": "op_1257790048", "stage": 0}
// {"group": 6, "id": 16, "lanes": [0, 4, 5, 6], "offset": 5, "op_kind": "arith.subf", "op_ref": "op_1257790256", "stage": 0}
// {"group": 2, "id": 17, "lanes": [2, 3, 4, 5], "offset": 0, "op_kind": "math.exp2", "op_ref": "op_1257791312", "stage": 1}
// {"group": 6, "id": 18, "lanes": [1, 2, 4, 5], "offset": 1, "op_kind": "arith.truncf", "op_ref": "op_1257786896", "stage": 1}
// {"group": 6, "id": 19, "lanes": [1], "offset": 2, "op_kind": "ttng.tmem_alloc", "op_ref": "op_1257813712", "stage": 1}
// {"group": 6, "id": 20, "lanes": [4], "offset": 0, "op_kind": "ttg.memdesc_trans", "op_ref": "op_1257813920", "stage": 0}
// {"group": 2, "id": 21, "lanes": [6], "offset": 0, "op_kind": "ttng.tc_gen5_mma", "op_ref": "op_1257069456", "stage": 0}
// {"group": 6, "id": 22, "lanes": [7], "offset": 3, "op_kind": "ttng.tc_gen5_mma", "op_ref": "op_1257071248", "stage": 1}
// {"group": 2, "id": 23, "lanes": [0, 1, 6, 7], "offset": 4, "op_kind": "tt.expand_dims", "op_ref": "op_1257816880", "stage": 0}
// {"group": 1, "id": 24, "lanes": [0, 1, 2, 3], "offset": 5, "op_kind": "tt.broadcast", "op_ref": "op_1257802688", "stage": 0}
// {"group": 1, "id": 25, "lanes": [0, 1, 2, 3], "offset": 5, "op_kind": "ttng.tmem_load", "op_ref": "op_1257067456", "stage": 0}
// {"group": 1, "id": 26, "lanes": [0, 1, 2, 3], "offset": 0, "op_kind": "arith.subf", "op_ref": "op_1257768320", "stage": 1}
// {"group": 1, "id": 27, "lanes": [0, 1, 2, 3], "offset": 1, "op_kind": "arith.mulf", "op_ref": "op_1257768576", "stage": 1}
// {"group": 6, "id": 28, "lanes": [3, 4, 6, 7], "offset": 2, "op_kind": "arith.truncf", "op_ref": "op_1257657136", "stage": 1}
// {"group": 1, "id": 29, "lanes": [1], "offset": 2, "op_kind": "ttg.local_alloc", "op_ref": "op_1257768992", "stage": 1}
// {"group": 1, "id": 30, "lanes": [1], "offset": 2, "op_kind": "ttg.memdesc_trans", "op_ref": "op_1257769200", "stage": 1}
// {"group": 6, "id": 31, "lanes": [2], "offset": 2, "op_kind": "ttng.tc_gen5_mma", "op_ref": "op_1256955904", "stage": 1}
// {"group": 4, "id": 32, "lanes": [0, 1, 2, 3], "offset": 4, "op_kind": "ttng.tmem_load", "op_ref": "op_1257068128", "stage": 1}
// {"group": 4, "id": 33, "lanes": [0, 1, 2, 3], "offset": 5, "op_kind": "arith.truncf", "op_ref": "op_1257793168", "stage": 1}
// {"group": 6, "id": 34, "lanes": [1, 2, 5, 7], "offset": 5, "op_kind": "ttg.convert_layout", "op_ref": "op_1257823280", "stage": 1}
// {"group": 7, "id": 35, "lanes": [0], "offset": 5, "op_kind": "tt.descriptor_reduce", "op_ref": "op_1257068768", "stage": 1}
// {"group": 7, "id": 36, "lanes": [1], "offset": 4, "op_kind": "ttng.tc_gen5_mma", "op_ref": "op_1256951024", "stage": 1}

// MANUAL: select CUDA instructions and data layouts.
// MANUAL: implement allocations and synchronization from reviewed plans.
