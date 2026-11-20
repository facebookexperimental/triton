import json

from triton.tools import amd_memory_width


def test_analyze_amdgcn_memory_width_reports_widths_and_merge_candidates():
    assembly = """\
.loc 1 42 3 ; /tmp/kernel.py:42:3
buffer_load_dword v40, off, s[4:7], 0 offset:0
buffer_load_dword v41, off, s[4:7], 0 offset:4
buffer_load_dword v42, off, s[4:7], 0 offset:8
buffer_load_dword v43, off, s[4:7], 0 offset:12
buffer_store_dword v8, v5, s[8:11], 0 offset:16 glc
buffer_store_dword v9, v5, s[8:11], 0 offset:20 glc
buffer_load_dwordx4 v[44:47], off, s[4:7], 0
global_load_dwordx2 v[48:49], v6, off
flat_store_short v7, v10
buffer_load_mystery v12, off, s[4:7], 0
; spill: buffer_load_dword v99, off, s[4:7], 0
"""

    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    assert result["instruction_count"] == 10
    assert result["known_width_instruction_count"] == 9
    assert result["unknown_width_instruction_count"] == 1
    assert result["total_bytes_per_lane"] == 50
    assert result["width_histogram_bytes"] == {"2": 1, "4": 6, "8": 1, "16": 1}
    assert result["narrow_instruction_count"] == 8
    assert result["wide_instruction_count"] == 1
    assert result["unknown_opcodes"] == {"buffer_load_mystery": 1}
    assert result["merge_opportunity_count"] == 2
    assert result["estimated_static_issue_reduction"] == 4

    load, store = result["merge_opportunities"]
    assert load["target_opcode"] == "buffer_load_dwordx4"
    assert (load["register_start"], load["register_end"]) == (40, 43)
    assert (load["offset_start"], load["offset_end"]) == (0, 12)
    assert load["source_location"] == "kernel.py:42:3"
    assert load["confidence"] == "needs_ir_proof"
    assert load["unproven_requirements"] == ["base_alignment", "bounds", "mask_equivalence"]
    assert store["target_opcode"] == "buffer_store_dwordx2"
    assert store["address_signature"].endswith("glc")


def test_analyze_amdgcn_memory_width_rejects_unsafe_scalar_groupings():
    assembly = """\
buffer_load_dword v1, off, s[4:7], 0 offset:0
buffer_load_dword v3, off, s[4:7], 0 offset:4
buffer_load_dword v4, off, s[4:7], 0 offset:12
v_add_u32 v9, v9, 1
buffer_load_dword v5, off, s[4:7], 0 offset:12
buffer_load_dword v6, off, s[8:11], 0 offset:16
buffer_load_dword v7, off, s[8:11], 0 offset:24
buffer_store_dword v10, v5, s[8:11], 0 offset:0 glc
buffer_store_dword v11, v5, s[8:11], 0 offset:4 slc
"""

    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    assert result["instruction_count"] == 8
    assert result["merge_opportunity_count"] == 0
    assert result["estimated_static_issue_reduction"] == 0


def test_analyze_amdgcn_memory_width_rejects_data_address_register_overlap():
    assembly = """\
buffer_load_dword v40, v40, s[4:7], 0 offset:0
buffer_load_dword v41, v40, s[4:7], 0 offset:4
"""

    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    assert result["instruction_count"] == 2
    assert result["merge_opportunity_count"] == 0


def test_analyze_amdgcn_memory_width_splits_six_scalar_loads_into_x4_and_x2():
    assembly = "\n".join(f"buffer_load_dword v{20 + index}, off, s[0:3], 0 offset:{4 * index}" for index in range(6))

    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    assert [item["target_opcode"] for item in result["merge_opportunities"]] == [
        "buffer_load_dwordx4",
        "buffer_load_dwordx2",
    ]
    assert result["estimated_static_issue_reduction"] == 4


def test_analyze_amdgcn_memory_width_reports_direct_to_lds_width_candidates():
    assembly = """\
.loc 1 1027 9 ; /tmp/mxfp8.py:1027:9
buffer_load_dword v131, s[44:47], 0 offen lds
buffer_load_dwordx4 v[20:23], off, s[0:3], 0
buffer_load_dword v132, off, s[0:3], 0
"""

    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    assert result["direct_to_lds_opportunity_count"] == 1
    assert result["potential_widening_count"] == 1
    candidate = result["direct_to_lds_opportunities"][0]
    assert candidate["opcode"] == "buffer_load_dword"
    assert candidate["target_opcode"] == "buffer_load_dwordx4"
    assert candidate["source_location"] == "mxfp8.py:1027:9"
    assert candidate["unproven_requirements"] == [
        "source_contiguity",
        "destination_contiguity",
        "base_alignment",
        "bounds",
        "useful_bytes",
    ]


def test_ttgir_evidence_preserves_candidate_with_additional_lane_data():
    assembly = "buffer_load_dword v1, s[0:3], 0 offen lds ; /tmp/kernel.py:12:3\n"
    ttgir = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
%0 = amdg.buffer_load_to_local %ptr[%offsets] into %dst : !tt.ptr<i8>[tensor<1x8xi32, #blocked>] -> <1x8xi8> loc(#loc1)
#loc1 = loc("/tmp/kernel.py":12:3)
"""
    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    amd_memory_width._add_ttgir_evidence(result, ttgir)

    assert result["direct_to_lds_ir_lane_saturated_count"] == 0
    assert result["direct_to_lds_unresolved_count"] == 1
    assert "ir_classification" not in result["direct_to_lds_opportunities"][0]


def test_ttgir_does_not_apply_kernel_wide_lane_saturation():
    assembly = "buffer_load_dword v1, s[0:3], 0 offen lds ; /tmp/kernel.py:12:3\n"
    ttgir = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
%0 = amdg.buffer_load_to_local %ptr[%offsets] into %dst : !tt.ptr<i8>[tensor<1x4xi32, #blocked>] -> <1x4xi8> loc(#loc1)
#loc1 = loc("/tmp/kernel.py":99:3)
"""
    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    amd_memory_width._add_ttgir_evidence(result, ttgir)

    assert result["direct_to_lds_ir_lane_saturated_count"] == 0
    assert result["direct_to_lds_unresolved_count"] == 1


def test_ttgir_reports_explicit_contiguity_below_lane_ownership():
    assembly = """\
.loc 1 42 3 ; /tmp/kernel.py:42:3
.Lmain:
    buffer_load_dwordx2 v[0:1], v0, s[0:3], 0 offen
    buffer_load_dwordx2 v[2:3], v1, s[0:3], 0 offen
    v_mfma_f32_16x16x32_f16 a[0:3], v[0:1], v[2:3], a[0:3]
    s_cbranch_scc0 .Lmain
"""
    ttgir = """\
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0]], warp = [[0, 8]], block = []}>
%0 = amdg.buffer_load %ptr[%offsets] {contiguity = 4 : i32, tlx.preserve_layout} : !tt.ptr<f16> -> tensor<64x32xf16, #linear> loc(#loc3)
%1 = amdg.buffer_load %ptr[%offsets] {contiguity = 4 : i32, tlx.preserve_layout} : !tt.ptr<f16> -> tensor<64x32xf16, #linear> loc(#loc3)
%2 = amdg.buffer_load %ptr[%offsets] {contiguity = 8 : i32, tlx.preserve_layout} : !tt.ptr<f16> -> tensor<64x32xf16, #linear> loc(#loc4)
#loc1 = loc("/tmp/kernel.py":42:3)
#loc2 = loc("/tmp/caller.py":9:2)
#loc3 = loc(callsite(#loc1 at #loc2))
#loc4 = loc("/tmp/kernel.py":51:7)
"""
    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    amd_memory_width._add_ttgir_evidence(result, ttgir)

    assert result["explicit_contiguity_cap_count"] == 1
    assert result["explicit_contiguity_cap_static_occurrences"] == 2
    assert result["explicit_contiguity_hot_loop_cap_count"] == 1
    assert result["explicit_contiguity_unresolved_count"] == 1
    assert result["explicit_contiguity_already_wide_count"] == 0
    assert result["potential_widening_count"] == 1
    candidate = result["explicit_contiguity_caps"][0]
    assert candidate["source_location"] == "kernel.py:42:3"
    assert candidate["elements_per_lane"] == 8
    assert candidate["explicit_contiguity"] == 4
    assert candidate["candidate_contiguity"] == 8
    assert candidate["explicit_bytes_per_lane"] == 8
    assert candidate["candidate_bytes_per_lane"] == 16
    assert candidate["static_occurrences"] == 2
    assert candidate["corroborated_by_hot_loop_headroom"]
    assert candidate["corroborated_by_narrow_isa"]
    assert candidate["narrow_load_instruction_count"] == 2
    assert candidate["confidence"] == "needs_address_proof"


def test_ttgir_contiguity_uses_contiguous_layout_ownership():
    assembly = "buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 offen ; /tmp/kernel.py:42:3\n"
    ttgir = """\
#blocked = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
%0 = amdg.buffer_load %ptr[%offsets] {contiguity = 4 : i32, tlx.preserve_layout} : !tt.ptr<f16> -> tensor<64x32xf16, #blocked> loc(#loc1)
#loc1 = loc("/tmp/kernel.py":42:3)
"""
    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    amd_memory_width._add_ttgir_evidence(result, ttgir)

    assert result["explicit_contiguity_cap_count"] == 0


def test_ttgir_linear_contiguity_excludes_broadcast_and_other_dimensions():
    ttgir = """\
#linear = #ttg.linear<{register = [[0, 0], [0, 1], [0, 2], [1, 0], [0, 4]], lane = [[2, 0]], warp = [], block = []}>
"""

    layout = amd_memory_width._ttgir_layout_info(ttgir.splitlines())["#linear"]

    assert layout["contiguous_elements_per_lane"] == 4
    assert layout["last_dimension_contiguous_elements_per_lane"] == 4


def test_ttgir_correlates_contiguity_caps_by_source_location():
    assembly = """\
.loc 1 42 3 ; /tmp/kernel.py:42:3
buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 offen
.loc 1 51 7 ; /tmp/kernel.py:51:7
buffer_load_dwordx2 v[4:5], v1, s[0:3], 0 offen
"""
    ttgir = """\
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0]], warp = [[0, 8]], block = []}>
%0 = amdg.buffer_load %ptr[%offsets] {contiguity = 4 : i32, tlx.preserve_layout} : !tt.ptr<f16> -> tensor<64x32xf16, #linear> loc(#loc1)
%1 = amdg.buffer_load %ptr[%offsets] {contiguity = 4 : i32, tlx.preserve_layout} : !tt.ptr<f16> -> tensor<64x32xf16, #linear> loc(#loc2)
#loc1 = loc("/tmp/kernel.py":42:3)
#loc2 = loc("/tmp/kernel.py":51:7)
"""
    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    amd_memory_width._add_ttgir_evidence(result, ttgir)

    by_source = {candidate["source_location"]: candidate for candidate in result["explicit_contiguity_caps"]}
    assert not by_source["kernel.py:42:3"]["corroborated_by_narrow_isa"]
    assert by_source["kernel.py:51:7"]["corroborated_by_narrow_isa"]
    assert result["explicit_contiguity_unresolved_count"] == 1


def test_ttgir_labels_kernel_level_isa_corroboration():
    assembly = "buffer_load_dwordx2 v[0:1], v0, s[0:3], 0 offen ; /tmp/other.py:7:1\n"
    ttgir = """\
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0]], warp = [[0, 8]], block = []}>
%0 = amdg.buffer_load %ptr[%offsets] {contiguity = 4 : i32, tlx.preserve_layout} : !tt.ptr<f16> -> tensor<64x32xf16, #linear> loc(#loc1)
#loc1 = loc("/tmp/kernel.py":42:3)
"""
    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    amd_memory_width._add_ttgir_evidence(result, ttgir)

    candidate = result["explicit_contiguity_caps"][0]
    assert candidate["corroborated_by_narrow_isa"]
    assert candidate["isa_corroboration_scope"] == "kernel"


def test_ttgir_reports_cross_lane_packed_byte_words():
    assembly = "buffer_load_ubyte v0, v1, s[0:3], 0 offen ; /tmp/gfx950.py:100:3\n"
    ttgir = """\
#linear = #ttg.linear<{register = [[32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[16, 0], [0, 0]], block = []}>
%0 = amdg.buffer_load %ptr[%offsets] : !tt.ptr<i8> -> tensor<128x4xi8, #linear> loc(#loc1)
#loc1 = loc("/tmp/gfx950.py":100:3)
"""
    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    amd_memory_width._add_ttgir_evidence(result, ttgir)

    assert result["packed_byte_load_candidate_count"] == 1
    assert result["packed_byte_load_unresolved_count"] == 1
    candidate = result["packed_byte_load_candidates"][0]
    assert candidate["logical_shape"] == [128, 4]
    assert candidate["classification"] == "cross_lane_packed_byte_word"
    assert candidate["isa_corroboration_scope"] == "source_location"
    assert candidate["corroborated_by_scalar_byte_isa"]


def test_ttgir_does_not_rank_contiguity_cap_when_isa_is_already_wide():
    assembly = "buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 offen\n"
    ttgir = """\
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0]], lane = [[8, 0]], warp = [[0, 8]], block = []}>
%0 = amdg.buffer_load %ptr[%offsets] {contiguity = 4 : i32, tlx.preserve_layout} : !tt.ptr<f16> -> tensor<64x32xf16, #linear> loc(#loc1)
#loc1 = loc("/tmp/kernel.py":42:3)
"""
    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    amd_memory_width._add_ttgir_evidence(result, ttgir)

    assert result["explicit_contiguity_cap_count"] == 1
    assert result["explicit_contiguity_unresolved_count"] == 0
    assert result["explicit_contiguity_already_wide_count"] == 1
    assert result["potential_widening_count"] == 0
    assert not result["explicit_contiguity_caps"][0]["corroborated_by_narrow_isa"]


def test_ttgir_reports_narrow_store_below_linear_layout_ownership():
    assembly = "buffer_store_short v0, v1, s[0:3], 0 offen ; /tmp/kernel.py:42:3\n"
    ttgir = """\
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[1, 0]], warp = [[8, 0]], block = []}>
amdg.buffer_store %value, %ptr[%offsets] : !tt.ptr<f16> -> tensor<32x32xf16, #linear> loc(#loc1)
#loc1 = loc("/tmp/kernel.py":42:3)
"""
    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    amd_memory_width._add_ttgir_evidence(result, ttgir)

    assert result["store_width_candidate_count"] == 1
    assert result["store_width_unresolved_count"] == 1
    candidate = result["store_width_candidates"][0]
    assert candidate["source_location"] == "kernel.py:42:3"
    assert candidate["contiguous_elements_per_lane"] == 8
    assert candidate["candidate_bytes_per_lane"] == 16
    assert candidate["narrow_store_instruction_count"] == 1
    assert candidate["corroborated_by_narrow_isa"]


def test_ttgir_marks_mfma_store_as_requiring_layout_conversion():
    assembly = "buffer_store_short v0, v1, s[0:3], 0 offen\n"
    ttgir = """\
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
amdg.buffer_store %value, %ptr[%offsets] : !tt.ptr<f16> -> tensor<32x32xf16, #mma> loc(#loc1)
#loc1 = loc("/tmp/kernel.py":42:3)
"""
    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    amd_memory_width._add_ttgir_evidence(result, ttgir)

    assert result["store_layout_conversion_count"] == 1
    assert result["store_width_unresolved_count"] == 0
    candidate = result["store_width_candidates"][0]
    assert candidate["candidate_bytes_per_lane"] is None
    assert candidate["classification"] == "requires_output_layout_conversion"


def test_ttgir_reports_blocked_store_without_layout_conversion():
    assembly = "buffer_store_dwordx4 v[0:3], v1, s[0:3], 0 offen ; /tmp/kernel.py:42:3\n"
    ttgir = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
amdg.buffer_store %value, %ptr[%offsets] : !tt.ptr<f16> -> tensor<32x32xf16, #blocked> loc(#loc1)
#loc1 = loc("/tmp/kernel.py":42:3)
"""
    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    amd_memory_width._add_ttgir_evidence(result, ttgir)

    candidate = result["store_width_candidates"][0]
    assert candidate["contiguous_elements_per_lane"] == 8
    assert candidate["candidate_bytes_per_lane"] == 16
    assert candidate["classification"] == "blocked_layout_store_width"
    assert result["store_layout_conversion_count"] == 0


def test_hot_loop_reports_reference_free_layout_headroom():
    assembly = """\
.Lmain:
    buffer_load_dwordx2 v[0:1], v0, s[0:3], 0 offen
    buffer_load_dwordx2 v[2:3], v1, s[0:3], 0 offen
    buffer_load_dwordx4 v[4:7], v2, s[4:7], 0 offen
    v_mfma_f32_16x16x32_f16 a[0:3], v[0:1], v[2:3], a[0:3]
    s_cbranch_scc0 .Lmain
.Ltail:
    v_mfma_f32_16x16x32_f16 a[0:3], v[0:1], v[2:3], a[0:3]
    v_mfma_f32_16x16x32_f16 a[4:7], v[0:1], v[2:3], a[4:7]
    s_cbranch_scc0 .Ltail
"""

    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    hot_loop = result["hot_loop_load_width"]
    assert hot_loop["found"]
    assert hot_loop["mfma_count"] == 1
    assert hot_loop["load_width_histogram_bytes"] == {"8": 2, "16": 1}
    assert hot_loop["load_issue_count"] == 3
    assert hot_loop["useful_bytes_per_lane"] == 32
    assert hot_loop["minimum_16byte_issue_count"] == 2
    assert hot_loop["excess_issue_count"] == 1
    assert hot_loop["issue_efficiency"] == 2 / 3
    assert hot_loop["requires_layout_redistribution"]


def test_analyze_amdgcn_memory_width_understands_format_channels():
    assembly = """\
buffer_load_format_x v1, off, s[0:3], 0 lds
buffer_load_format_xy v[2:3], off, s[0:3], 0
buffer_load_format_d16_xyz v[4:5], off, s[0:3], 0
buffer_store_format_xyzw v[6:9], off, s[0:3], 0
"""

    result = amd_memory_width.analyze_amdgcn_memory_width(assembly)

    assert result["width_histogram_bytes"] == {"4": 1, "6": 1, "8": 1, "16": 1}
    assert result["unknown_width_instruction_count"] == 0
    assert result["direct_to_lds_opportunities"][0]["target_opcode"] == "buffer_load_format_xyzw"


def test_main_emits_json_for_amdgcn_input(tmp_path, capsys):
    path = tmp_path / "kernel.amdgcn"
    path.write_text("buffer_load_dwordx4 v[0:3], off, s[0:3], 0\n")

    assert amd_memory_width.main([str(path), "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["width_histogram_bytes"] == {"16": 1}
    assert report["merge_opportunity_count"] == 0


def test_main_disassembles_hsaco_with_explicit_objdump(tmp_path, capsys, monkeypatch):
    path = tmp_path / "kernel.hsaco"
    path.write_bytes(b"code-object")

    def fake_check_output(command, text):
        assert command == ["/fake/llvm-objdump", "-d", "-l", "--arch=amdgcn", str(path)]
        assert text is True
        return "buffer_store_dwordx2 v[0:1], off, s[0:3], 0\n"

    monkeypatch.setattr(amd_memory_width.subprocess, "check_output", fake_check_output)

    assert amd_memory_width.main([str(path), "--objdump", "/fake/llvm-objdump", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["width_histogram_bytes"] == {"8": 1}


def test_main_summarizes_indexed_dump_directory(tmp_path, capsys):
    dump = tmp_path / "dump"
    mm_dir = dump / "mmhash"
    mxfp_dir = dump / "mxfphash"
    mm_dir.mkdir(parents=True)
    mxfp_dir.mkdir(parents=True)
    (mm_dir / "mm.amdgcn").write_text("buffer_load_dwordx4 v[0:3], off, s[0:3], 0\n")
    (mxfp_dir / "mxfp.amdgcn").write_text("buffer_load_dword v1, s[0:3], 0 offen lds ; /tmp/mxfp.py:12:3\n")
    (mxfp_dir / "mxfp.ttgir").write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
%0 = amdg.buffer_load_to_local %ptr[%offsets] into %dst : !tt.ptr<i8>[tensor<1x4xi32, #blocked>] -> <1x4xi8> loc(#loc1)
#loc1 = loc("/tmp/mxfp.py":12:3)
""")
    index = {
        "cases": [{"case": "mm/gfx950/fp16/1x1x1|fwd"}, {"case": "mm_mxfp8/gfx950/fp8/1x1x1|fwd"}],
        "compiles": [
            {"dump": "mmhash", "name": "mm_kernel", "case": "mm/gfx950/fp16/1x1x1|fwd"},
            {"dump": "mxfphash", "name": "mxfp_kernel", "case": "mm_mxfp8/gfx950/fp8/1x1x1|fwd"},
        ],
    }
    (tmp_path / "index.json").write_text(json.dumps(index))

    assert amd_memory_width.main([str(tmp_path), "--json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["indexed_case_count"] == 2
    assert result["overall"]["kernel_count"] == 2
    assert result["overall"]["fully_wide_kernel_count"] == 1
    assert result["overall"]["direct_to_lds_opportunity_count"] == 1
    assert result["overall"]["direct_to_lds_ir_lane_saturated_count"] == 1
    assert result["overall"]["direct_to_lds_unresolved_count"] == 0
    assert result["overall"]["explicit_contiguity_cap_count"] == 0
    assert result["by_family"]["mm"]["width_histogram_bytes"] == {"16": 1}
    assert result["by_family"]["mm_mxfp8"]["width_histogram_bytes"] == {"4": 1}
    assert result["top_potential_widening"] == []
    assert result["top_explicit_contiguity_caps"] == []
    assert result["top_store_width_candidates"] == []
