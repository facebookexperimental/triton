from triton.language import core

__all__ = ["_mul_f32x2", "_sub_f32x2", "_fma_f32x2", "_reduce_fadd2", "tanh_approx_fp32"]


@core.builtin
def _mul_f32x2(a, b, _semantic=None):
    return core.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mul.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=core.float32,
        is_pure=True,
        pack=2,
        _semantic=_semantic,
    )


@core.builtin
def _sub_f32x2(a, b, _semantic=None):
    return core.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            sub.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=core.float32,
        is_pure=True,
        pack=2,
        _semantic=_semantic,
    )


@core.builtin
def _fma_f32x2(a, b, c, _semantic=None):
    return core.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc, rd;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mov.b64 rc, { $6, $7 };
            fma.rn.f32x2 rd, ra, rb, rc;
            mov.b64 { $0, $1 }, rd;
        }
        """,
        "=r,=r,r,r,r,r,r,r",
        [a, b, c],
        dtype=core.float32,
        is_pure=True,
        pack=2,
        _semantic=_semantic,
    )


@core.builtin
def _reduce_fadd2(p0a, p1a, p0b, p1b, _semantic=None):
    return core.inline_asm_elementwise(
        """
        {
            .reg .b64 rc, ra, rb;
            mov.b64 ra, { $2, $4 };
            mov.b64 rb, { $3, $5 };
            add.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [p0a, p0b, p1a, p1b],
        dtype=[core.float32, core.float32],
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


@core.builtin
def tanh_approx_fp32(x, _semantic=None):
    return core.inline_asm_elementwise(
        """
        tanh.approx.f32 $0, $1;
        """,
        "=r,r",
        [x],
        dtype=core.float32,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )
