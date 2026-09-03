import torch

import triton.tlx.ops as ops
from triton.tlx.ops._catalog import OpSpec


def test_backend_without_accepts_keeps_arbitrary_strides(monkeypatch):
    spec = OpSpec(
        op="mm",
        arch="test",
        variant="strided",
        impl="unused",
        dtypes=frozenset({"float16"}),
    )
    sentinel = object()

    def fake_mm(a, b, *, space):
        assert space == "heuristic"
        return sentinel

    monkeypatch.setattr(ops, "impl_for", lambda op, arch: (fake_mm, spec))
    a = torch.empty((8, 1894), dtype=torch.float16)[:, :16]
    b = torch.empty((16, 24), dtype=torch.float16)

    assert ops.mm(a, b, arch="test") is sentinel
