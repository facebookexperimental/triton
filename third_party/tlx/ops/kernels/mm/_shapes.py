from __future__ import annotations

# Entries are [M, N, K, a_strides, b_strides, dtype]. A is (M, K), B is (K, N);
# row-major is (trailing, 1), column-major is (1, leading).

SYNTHETIC: list[list] = [
    [256, 256, 256, (256, 1), (256, 1), "fp16"],
    [1024, 1024, 1024, (1024, 1), (1024, 1), "fp16"],
    [2048, 512, 1024, (1024, 1), (512, 1), "fp16"],
    [512, 4096, 1024, (1024, 1), (1, 1024), "fp16"],
    [1024, 2048, 512, (1, 1024), (2048, 1), "fp16"],
    [2048, 2048, 2048, (1, 2048), (1, 2048), "fp16"],
    [136, 256, 128, (128, 1), (256, 1), "fp16"],
    [1000, 1000, 1024, (1024, 1), (1000, 1), "fp16"],
    [1000, 1000, 200, (200, 1), (1000, 1), "fp16"],
    [256, 256, 16384, (16384, 1), (256, 1), "fp16"],
    [64, 4096, 4096, (4096, 1), (4096, 1), "fp16"],
    [256, 256, 256, (256, 1), (256, 1), "bf16"],
    [1024, 1024, 1024, (1024, 1), (1024, 1), "bf16"],
    [2048, 512, 1024, (1024, 1), (512, 1), "bf16"],
    [512, 4096, 1024, (1024, 1), (1, 1024), "bf16"],
    [1024, 2048, 512, (1, 1024), (2048, 1), "bf16"],
    [2048, 2048, 2048, (1, 2048), (1, 2048), "bf16"],
    [136, 256, 128, (128, 1), (256, 1), "bf16"],
    [1000, 1000, 1024, (1024, 1), (1000, 1), "bf16"],
    [1000, 1000, 200, (200, 1), (1000, 1), "bf16"],
    [256, 256, 16384, (16384, 1), (256, 1), "bf16"],
    [64, 4096, 4096, (4096, 1), (4096, 1), "bf16"],
]

# The compute-bound shapes the perf suite gates on for this arch.
SM100_FOCUS: list[list] = [
    [8192, 8192, 8192, (8192, 1), (8192, 1), "fp16"],
    [8192, 8192, 8192, (8192, 1), (8192, 1), "bf16"],
    [8192, 8192, 1024, (1024, 1), (8192, 1), "fp16"],
    [8192, 8192, 1024, (1024, 1), (8192, 1), "bf16"],
    [8192, 8192, 16384, (16384, 1), (8192, 1), "fp16"],
    [8192, 8192, 16384, (16384, 1), (8192, 1), "bf16"],
    [8192, 8192, 8192, (8192, 1), (1, 8192), "fp16"],
    [8192, 8192, 8192, (8192, 1), (1, 8192), "bf16"],
]


def _union(*lists: list[list]) -> list[list]:
    seen, out = set(), []
    for shapes in lists:
        for shape in shapes:
            key = tuple(shape)
            if key not in seen:
                seen.add(key)
                out.append(shape)
    return out


ALL: list[list] = _union(SYNTHETIC, SM100_FOCUS)


def operand(rows, cols, strides, dtype, device="cuda"):
    """A (rows, cols) tensor whose strides are exactly `strides`.

    Recorded strides carry three things a row/column-major flag cannot:
    a leading stride wider than the row (a slice of a padded buffer),
    stride 0 (a broadcast operand), and which of the two dims is contiguous.
    """
    import torch

    s0, s1 = strides
    if s0 == 0:  # broadcast down the rows
        # At rows == 1 the expand is a no-op and stride(0) stays `cols`. That is
        # not a miss: a stride on a dim of extent 1 addresses nothing, so the
        # element layout is identical either way.
        return torch.randn((1, cols), device=device, dtype=dtype).expand(rows, cols)
    if s1 == 1:  # row-major, possibly padded to s0 >= cols
        return torch.randn((rows, s0), device=device, dtype=dtype)[:, :cols]
    if s0 == 1:  # column-major, possibly padded to s1 >= rows
        return torch.randn((cols, s1), device=device, dtype=dtype)[:, :rows].T
    raise ValueError(f"unsupported strides {strides} for ({rows}, {cols})")


def label(M, N, K, a_strides, b_strides, dtype) -> str:
    """The report's input column, in the capture's own notation."""
    strides = f"[[{a_strides[0]}, {a_strides[1]}], [{b_strides[0]}, {b_strides[1]}]]"
    return (f"((), {{'dtype': '{dtype}', 'strides': '{strides}', "
            f"'M': '{M}', 'N': '{N}', 'K': '{K}'}})")


def flops(M, N, K):
    return 2 * M * N * K
