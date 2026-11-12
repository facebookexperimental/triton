import tlx_ops_bench_ir_diff as ir_diff

BASE_TTGIR = """#loc = loc("k.py":1:0)
module {
  tt.func public @k(%X: !tt.ptr<f32> loc("X"(#loc)), %N: i32 loc("N"(#loc))) {
    %mu = arith.constant dense<2.560000e+02> : tensor<32xf32> loc(#loc1)
    %s = arith.divf %x, %mu : tensor<32xf32> loc(callsite(#loc5 at #loc156))
    tt.return loc(#loc2)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("k.py":3:4)
"""

RENAMED_TTGIR = """#loc = loc("other.py":9:0)
module {
  tt.func public @k(%X: !tt.ptr<f32> loc("X"(#loc)), %N: i32 loc("N"(#loc))) {
    %cst_7 = arith.constant dense<2.560000e+02> : tensor<32xf32> loc(#loc8)
    %s_12 = arith.divf %x_3, %cst_7 : tensor<32xf32> loc(callsite(#loc11 at #loc158))
    tt.return loc(#loc9)
  } loc(#loc)
} loc(#loc)
#loc8 = loc("other.py":30:4)
"""

REWRITTEN_TTGIR = """module {
  tt.func public @k(%X: !tt.ptr<f32>, %N: i32) {
    %c = arith.constant 2.560000e+02 : f32
    %one = arith.constant 1.000000e+00 : f32
    %r = tt.approx_divf %one, %c : f32
    %rs = tt.splat %r : f32 -> tensor<32xf32>
    %s = arith.mulf %x, %rs : tensor<32xf32>
    tt.return
  }
}
"""


def test_location_and_value_name_churn_is_not_a_code_change():
    base = ir_diff.normalize("ttgir", BASE_TTGIR)
    assert base == ir_diff.normalize("ttgir", RENAMED_TTGIR)
    rewritten = ir_diff.normalize("ttgir", REWRITTEN_TTGIR)
    assert base != rewritten
    assert dict(ir_diff.deltas(ir_diff.histogram("ttgir", base), ir_diff.histogram("ttgir", rewritten))) == {
        "arith.divf": -1, "arith.mulf": 1, "arith.constant": 1, "tt.approx_divf": 1, "tt.splat": 1
    }


def gcn(path, line, insn):
    return f"""\t.text
\t.file\t1 "{path}" "k.py"
\t.loc\t1 {line} 0
\tv_add_f32_e32 v1, v2, v3 ; {line}
\t{insn}
\t.section\t.debug_str,"MS",@progbits,1
\t.asciz\t"{path}" ; string offset=17
\t.text
\t.ident\t"Triton version 3.9.0.dev2026111{line % 10}"
"""


def ptx(path, line, insn):
    return f""".visible .entry k()
{{
\t.loc\t1 {line} 0, function_name $L__info_string0, inlined_at 1 {line * 4} 36 // standard.py:{line}:16
\tadd.f32 \t%f1, %f2, %f3; // k.py:{line}
\t{insn}
}}
\t.file\t1 "{path}/k.py"
\t.section\t.debug_str
\t{{
.b8 {line}
\t}}
"""


def test_source_paths_and_lines_in_debug_info_are_not_a_code_change():
    # The same kernel compiled from another install location or source line must hash and diff as unchanged, or a
    # trace between two wheel installs reports every kernel as changed.
    import tlx_codegen_snapshot as snapshot
    assert ir_diff.normalize("amdgcn", gcn("/a/venv", 3,
                                           "s_endpgm")) == ir_diff.normalize("amdgcn", gcn("/b/venv", 7, "s_endpgm"))
    assert ir_diff.normalize("amdgcn", gcn("/a/venv", 3,
                                           "s_endpgm")) != ir_diff.normalize("amdgcn", gcn("/a/venv", 3, "s_nop 0"))
    assert snapshot.normalize_ptx(ptx("/a/venv", 3, "ret;")) == snapshot.normalize_ptx(ptx("/b/venv", 7, "ret;"))
    assert snapshot.normalize_ptx(ptx("/a/venv", 3, "ret;")) != snapshot.normalize_ptx(ptx("/a/venv", 3, "exit;"))
