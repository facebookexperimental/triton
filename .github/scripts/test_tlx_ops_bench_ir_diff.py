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
