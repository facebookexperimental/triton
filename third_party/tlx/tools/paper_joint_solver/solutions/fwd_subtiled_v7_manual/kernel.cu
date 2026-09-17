/*
 * Deterministic SKC scaffold for expert manual CUDA lowering.
 * This file intentionally contains no executable kernel.
 * Complete and review the manifests before authoring CUDA.
 */
#error "Manual CUDA lowering is required; this scaffold is not executable"

// Twill IR sha256: fb2718496d24e6cb891ae0d69569750f7f50996b93a0a3c0b9a04a2c07d9fdba
// Scheduled instructions (schedule facts only):
// {"group": 1, "id": 0, "lanes": [0], "offset": 0, "op_kind": "arith.muli", "op_ref": "op_269860608", "stage": 0}
// {"group": 2, "id": 1, "lanes": [0], "offset": 0, "op_kind": "arith.addi", "op_ref": "op_269860976", "stage": 0}
// {"group": 0, "id": 2, "lanes": [0], "offset": 0, "op_kind": "tt.descriptor_load", "op_ref": "op_269446992", "stage": 0}
// {"group": 0, "id": 3, "lanes": [0], "offset": 1, "op_kind": "tt.descriptor_load", "op_ref": "op_269870224", "stage": 0}
// {"group": 3, "id": 4, "lanes": [0], "offset": 1, "op_kind": "ttg.local_alloc", "op_ref": "op_269844288", "stage": 1}
// {"group": 4, "id": 5, "lanes": [0], "offset": 0, "op_kind": "ttg.local_alloc", "op_ref": "op_269844496", "stage": 0}
// {"group": 5, "id": 6, "lanes": [0], "offset": 0, "op_kind": "ttg.memdesc_trans", "op_ref": "op_269862480", "stage": 0}
// {"group": 6, "id": 7, "lanes": [0], "offset": 0, "op_kind": "ttng.tc_gen5_mma", "op_ref": "op_269136640", "stage": 0}
// {"group": 7, "id": 8, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "ttng.tmem_load", "op_ref": "op_269134912", "stage": 0}
// {"group": 7, "id": 9, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "tt.reduce", "op_ref": "op_269857904", "stage": 0}
// {"group": 7, "id": 10, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "arith.mulf", "op_ref": "op_269847264", "stage": 0}
// {"group": 7, "id": 11, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "arith.maxnumf", "op_ref": "op_269852144", "stage": 0}
// {"group": 7, "id": 12, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "arith.subf", "op_ref": "op_269847456", "stage": 0}
// {"group": 2, "id": 13, "lanes": [1, 2, 3, 4], "offset": 2, "op_kind": "math.exp2", "op_ref": "op_269847712", "stage": 0}
// {"group": 7, "id": 14, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "arith.mulf", "op_ref": "op_269853776", "stage": 0}
// {"group": 7, "id": 15, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "tt.expand_dims", "op_ref": "op_269882064", "stage": 0}
// {"group": 7, "id": 16, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "tt.broadcast", "op_ref": "op_269882272", "stage": 0}
// {"group": 2, "id": 17, "lanes": [0, 1, 2, 5], "offset": 2, "op_kind": "arith.subf", "op_ref": "op_269882480", "stage": 0}
// {"group": 7, "id": 18, "lanes": [0, 1, 2, 3], "offset": 0, "op_kind": "math.exp2", "op_ref": "op_269882736", "stage": 1}
// {"group": 2, "id": 19, "lanes": [1, 2, 3, 5], "offset": 1, "op_kind": "tt.reduce", "op_ref": "op_269883008", "stage": 1}
// {"group": 7, "id": 20, "lanes": [0, 1, 2, 3], "offset": 1, "op_kind": "arith.truncf", "op_ref": "op_269736256", "stage": 1}
// {"group": 8, "id": 21, "lanes": [0], "offset": 2, "op_kind": "ttng.tmem_alloc", "op_ref": "op_269887616", "stage": 1}
// {"group": 9, "id": 22, "lanes": [0, 1, 2, 3], "offset": 0, "op_kind": "tt.expand_dims", "op_ref": "op_269887760", "stage": 1}
// {"group": 9, "id": 23, "lanes": [0, 1, 2, 3], "offset": 0, "op_kind": "ttg.convert_layout", "op_ref": "op_269888064", "stage": 1}
// {"group": 2, "id": 24, "lanes": [0, 4, 5, 6], "offset": 0, "op_kind": "tt.broadcast", "op_ref": "op_269888272", "stage": 1}
// {"group": 9, "id": 25, "lanes": [0, 1, 2, 4], "offset": 0, "op_kind": "ttng.tmem_load", "op_ref": "op_269885184", "stage": 1}
// {"group": 9, "id": 26, "lanes": [0, 1, 3, 5], "offset": 1, "op_kind": "arith.mulf", "op_ref": "op_269892272", "stage": 1}
// {"group": 2, "id": 27, "lanes": [1, 3, 4, 7], "offset": 2, "op_kind": "ttng.tmem_store", "op_ref": "op_269213280", "stage": 1}
// {"group": 9, "id": 28, "lanes": [0], "offset": 2, "op_kind": "ttng.tc_gen5_mma", "op_ref": "op_269142928", "stage": 1}
// {"group": 9, "id": 29, "lanes": [1, 4, 6, 7], "offset": 2, "op_kind": "arith.mulf", "op_ref": "op_269892944", "stage": 1}
// {"group": 9, "id": 30, "lanes": [1, 4, 6, 7], "offset": 2, "op_kind": "arith.addf", "op_ref": "op_269893200", "stage": 1}
// {"group": 1, "id": 31, "lanes": [0], "offset": 0, "op_kind": "ttng.tc_gen5_mma", "op_ref": "op_269144720", "stage": 0}
// {"group": 2, "id": 32, "lanes": [3, 5, 6, 7], "offset": 2, "op_kind": "ttng.tmem_load", "op_ref": "op_269862832", "stage": 0}
// {"group": 9, "id": 33, "lanes": [2, 3, 5, 6], "offset": 2, "op_kind": "tt.reduce", "op_ref": "op_269890736", "stage": 0}
// {"group": 9, "id": 34, "lanes": [2, 3, 5, 6], "offset": 2, "op_kind": "arith.mulf", "op_ref": "op_269868656", "stage": 0}
// {"group": 9, "id": 35, "lanes": [2, 3, 5, 6], "offset": 2, "op_kind": "arith.maxnumf", "op_ref": "op_269883184", "stage": 0}
// {"group": 9, "id": 36, "lanes": [2, 3, 5, 6], "offset": 2, "op_kind": "arith.subf", "op_ref": "op_269895280", "stage": 0}
// {"group": 9, "id": 37, "lanes": [4, 5, 6, 7], "offset": 2, "op_kind": "math.exp2", "op_ref": "op_269895536", "stage": 0}
// {"group": 9, "id": 38, "lanes": [1, 3, 4, 7], "offset": 2, "op_kind": "arith.mulf", "op_ref": "op_269868848", "stage": 0}
// {"group": 9, "id": 39, "lanes": [1, 2, 3, 5], "offset": 2, "op_kind": "tt.expand_dims", "op_ref": "op_269869104", "stage": 0}
// {"group": 9, "id": 40, "lanes": [1, 2, 3, 5], "offset": 2, "op_kind": "tt.broadcast", "op_ref": "op_269869312", "stage": 0}
// {"group": 2, "id": 41, "lanes": [2, 3, 6, 7], "offset": 2, "op_kind": "arith.subf", "op_ref": "op_269869520", "stage": 0}
// {"group": 2, "id": 42, "lanes": [1, 2, 3, 7], "offset": 0, "op_kind": "math.exp2", "op_ref": "op_269869776", "stage": 1}
// {"group": 2, "id": 43, "lanes": [0, 4, 6, 7], "offset": 1, "op_kind": "tt.reduce", "op_ref": "op_269884832", "stage": 1}
// {"group": 2, "id": 44, "lanes": [1, 2, 3, 7], "offset": 1, "op_kind": "arith.truncf", "op_ref": "op_269731104", "stage": 1}
// {"group": 9, "id": 45, "lanes": [4], "offset": 1, "op_kind": "ttng.tmem_alloc", "op_ref": "op_269902656", "stage": 1}
// {"group": 2, "id": 46, "lanes": [0, 1, 2, 7], "offset": 2, "op_kind": "tt.expand_dims", "op_ref": "op_269902800", "stage": 0}
// {"group": 9, "id": 47, "lanes": [1, 2, 5, 6], "offset": 2, "op_kind": "ttg.convert_layout", "op_ref": "op_269903008", "stage": 0}
// {"group": 7, "id": 48, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "tt.broadcast", "op_ref": "op_269903216", "stage": 0}
// {"group": 7, "id": 49, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "ttng.tmem_load", "op_ref": "op_269880528", "stage": 0}
// {"group": 7, "id": 50, "lanes": [0, 1, 2, 3], "offset": 0, "op_kind": "arith.mulf", "op_ref": "op_269908368", "stage": 1}
// {"group": 7, "id": 51, "lanes": [0, 1, 2, 3], "offset": 1, "op_kind": "ttng.tmem_store", "op_ref": "op_269217168", "stage": 1}
// {"group": 5, "id": 52, "lanes": [1], "offset": 1, "op_kind": "ttng.tc_gen5_mma", "op_ref": "op_269029376", "stage": 1}
// {"group": 9, "id": 53, "lanes": [1, 3, 6, 7], "offset": 2, "op_kind": "arith.mulf", "op_ref": "op_269908912", "stage": 0}
// {"group": 9, "id": 54, "lanes": [1, 3, 6, 7], "offset": 2, "op_kind": "arith.addf", "op_ref": "op_269909168", "stage": 1}

// MANUAL: select CUDA instructions and data layouts.
// MANUAL: implement allocations and synchronization from reviewed plans.
