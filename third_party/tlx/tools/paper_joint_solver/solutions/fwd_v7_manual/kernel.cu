/*
 * Deterministic SKC scaffold for expert manual CUDA lowering.
 * This file intentionally contains no executable kernel.
 * Complete and review the manifests before authoring CUDA.
 */
#error "Manual CUDA lowering is required; this scaffold is not executable"

// Twill IR sha256: 3c9486dc0b710d02545b2e636b39045d9e777fc3662bf1933d958093b0fc3acb
// Scheduled instructions (schedule facts only):
// {"group": 1, "id": 0, "lanes": [0], "offset": 0, "op_kind": "arith.muli", "op_ref": "op_264768288", "stage": 0}
// {"group": 2, "id": 1, "lanes": [0], "offset": 0, "op_kind": "arith.addi", "op_ref": "op_264768656", "stage": 0}
// {"group": 0, "id": 2, "lanes": [0], "offset": 0, "op_kind": "tt.descriptor_load", "op_ref": "op_264377440", "stage": 0}
// {"group": 0, "id": 3, "lanes": [0], "offset": 2, "op_kind": "tt.descriptor_load", "op_ref": "op_264376128", "stage": 0}
// {"group": 3, "id": 4, "lanes": [0], "offset": 0, "op_kind": "ttg.local_alloc", "op_ref": "op_264786080", "stage": 1}
// {"group": 4, "id": 5, "lanes": [0], "offset": 0, "op_kind": "ttg.local_alloc", "op_ref": "op_264778080", "stage": 0}
// {"group": 5, "id": 6, "lanes": [0], "offset": 0, "op_kind": "ttg.memdesc_trans", "op_ref": "op_264778416", "stage": 0}
// {"group": 5, "id": 7, "lanes": [1], "offset": 0, "op_kind": "ttng.tc_gen5_mma", "op_ref": "op_264065792", "stage": 0}
// {"group": 6, "id": 8, "lanes": [0, 1, 2, 3], "offset": 1, "op_kind": "ttng.tmem_load", "op_ref": "op_264785712", "stage": 0}
// {"group": 3, "id": 9, "lanes": [0, 1, 2, 3], "offset": 1, "op_kind": "tt.reduce", "op_ref": "op_264781840", "stage": 0}
// {"group": 7, "id": 10, "lanes": [0, 1, 2, 3], "offset": 1, "op_kind": "arith.mulf", "op_ref": "op_264781280", "stage": 0}
// {"group": 3, "id": 11, "lanes": [1, 2, 4, 5], "offset": 1, "op_kind": "arith.maxnumf", "op_ref": "op_264796880", "stage": 0}
// {"group": 3, "id": 12, "lanes": [1, 2, 4, 5], "offset": 2, "op_kind": "arith.subf", "op_ref": "op_264788320", "stage": 0}
// {"group": 6, "id": 13, "lanes": [0, 1, 2, 3], "offset": 2, "op_kind": "math.exp2", "op_ref": "op_264797328", "stage": 0}
// {"group": 7, "id": 14, "lanes": [0, 1, 4, 5], "offset": 1, "op_kind": "arith.mulf", "op_ref": "op_264797536", "stage": 0}
// {"group": 3, "id": 15, "lanes": [0, 1, 2, 3], "offset": 1, "op_kind": "tt.expand_dims", "op_ref": "op_264797984", "stage": 0}
// {"group": 3, "id": 16, "lanes": [0, 1, 3, 4], "offset": 1, "op_kind": "tt.broadcast", "op_ref": "op_264798192", "stage": 0}
// {"group": 7, "id": 17, "lanes": [0, 1, 4, 5], "offset": 2, "op_kind": "arith.subf", "op_ref": "op_264798576", "stage": 0}
// {"group": 3, "id": 18, "lanes": [1, 2, 4, 5], "offset": 2, "op_kind": "math.exp2", "op_ref": "op_264799632", "stage": 0}
// {"group": 3, "id": 19, "lanes": [1, 2, 6, 7], "offset": 0, "op_kind": "tt.reduce", "op_ref": "op_264797088", "stage": 1}
// {"group": 7, "id": 20, "lanes": [0, 2, 6, 7], "offset": 0, "op_kind": "arith.truncf", "op_ref": "op_264663600", "stage": 1}
// {"group": 8, "id": 21, "lanes": [0], "offset": 1, "op_kind": "ttng.tmem_alloc", "op_ref": "op_264807488", "stage": 1}
// {"group": 7, "id": 22, "lanes": [0, 3, 6, 7], "offset": 2, "op_kind": "tt.expand_dims", "op_ref": "op_264807632", "stage": 0}
// {"group": 7, "id": 23, "lanes": [0, 1, 6, 7], "offset": 2, "op_kind": "ttg.convert_layout", "op_ref": "op_264807840", "stage": 0}
// {"group": 7, "id": 24, "lanes": [3, 4, 5, 6], "offset": 2, "op_kind": "tt.broadcast", "op_ref": "op_264808048", "stage": 0}
// {"group": 3, "id": 25, "lanes": [0, 3, 6, 7], "offset": 2, "op_kind": "ttng.tmem_load", "op_ref": "op_264799856", "stage": 0}
// {"group": 7, "id": 26, "lanes": [3, 4, 5, 6], "offset": 0, "op_kind": "arith.mulf", "op_ref": "op_264800448", "stage": 1}
// {"group": 7, "id": 27, "lanes": [0, 1, 2, 3], "offset": 1, "op_kind": "ttng.tmem_store", "op_ref": "op_264145856", "stage": 1}
// {"group": 9, "id": 28, "lanes": [0], "offset": 1, "op_kind": "ttng.tc_gen5_mma", "op_ref": "op_264072080", "stage": 1}
// {"group": 7, "id": 29, "lanes": [0, 3, 6, 7], "offset": 1, "op_kind": "arith.mulf", "op_ref": "op_264810576", "stage": 1}
// {"group": 7, "id": 30, "lanes": [0, 3, 6, 7], "offset": 1, "op_kind": "arith.addf", "op_ref": "op_264810832", "stage": 1}

// MANUAL: select CUDA instructions and data layouts.
// MANUAL: implement allocations and synchronization from reviewed plans.
