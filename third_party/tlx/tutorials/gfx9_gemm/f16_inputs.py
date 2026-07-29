"""Deterministic input generation shared by the gfx9 f16 GEMM benchmarks."""

import torch

INPUT_MODES = ("normal", "hpl", "rand-int", "zero", "ones")
DEFAULT_INPUT_SEED = 0

_INPUT_INIT_CHUNK_ELEMENTS = 4 * 1024 * 1024
_HIPBLASLT_RNG_SEED_STRIDE = 0x9E3779B9
_UINT32_MAX = (1 << 32) - 1
_LOGICAL_RSHIFT_17_MASK = (1 << (64 - 17)) - 1


def _hipblaslt_random_u32(indices):
    """Return hipBLASLt's deterministic pseudo_random_device value per index."""
    state = indices * 1664525 + 1013904223
    for _ in range(3):
        logical_rshift_17 = (state >> 17) & _LOGICAL_RSHIFT_17_MASK
        state = state ^ (state << 13) ^ logical_rshift_17 ^ (state << 5)
    return state & _UINT32_MAX


def _make_hipblaslt_input(rows, cols, device, input_mode, seed, *, checkerboard_sign=False):
    """Build exact hipBLASLt HPL/rand_int data without large temporaries."""
    result = torch.empty((rows, cols), device=device, dtype=torch.float16)
    chunk_rows = max(1, _INPUT_INIT_CHUNK_ELEMENTS // cols)
    seed_offset = seed * _HIPBLASLT_RNG_SEED_STRIDE
    col_ids = None
    if checkerboard_sign:
        col_ids = torch.arange(cols, device=device, dtype=torch.int64)[None, :]

    for row_begin in range(0, rows, chunk_rows):
        row_end = min(rows, row_begin + chunk_rows)
        flat_begin = row_begin * cols
        flat_end = row_end * cols
        indices = torch.arange(flat_begin, flat_end, device=device, dtype=torch.int64)
        random_u32 = _hipblaslt_random_u32(indices + seed_offset)
        if input_mode == "hpl":
            values = random_u32.to(torch.float64) / float(_UINT32_MAX) - 0.5
        else:
            values = random_u32.remainder(5) - 2
            if checkerboard_sign:
                row_ids = torch.arange(row_begin, row_end, device=device, dtype=torch.int64)[:, None]
                negate = ((row_ids ^ col_ids) & 1) == 0
                values = values.reshape(row_end - row_begin, cols)
                values = torch.where(negate, -values, values)
        result[row_begin:row_end].copy_(values.reshape(row_end - row_begin, cols))
    return result


def _make_input_storage(rows, cols, device, input_mode, seed, generator, *, is_b=False):
    if input_mode == "normal":
        return torch.randn(
            (rows, cols),
            device=device,
            dtype=torch.float16,
            generator=generator,
        )
    if input_mode in {"hpl", "rand-int"}:
        return _make_hipblaslt_input(
            rows,
            cols,
            device,
            input_mode,
            seed,
            checkerboard_sign=is_b and input_mode == "rand-int",
        )
    if input_mode == "zero":
        return torch.zeros((rows, cols), device=device, dtype=torch.float16)
    if input_mode == "ones":
        return torch.ones((rows, cols), device=device, dtype=torch.float16)
    raise ValueError(f"unsupported input mode: {input_mode}")


def make_inputs(M, N, K, device, b_layout, input_mode="normal", seed=DEFAULT_INPUT_SEED):
    """Create an MxK A and KxN B using one of the shared f16 input modes."""
    generator = None
    if input_mode == "normal":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    a = _make_input_storage(M, K, device, input_mode, seed, generator)
    if b_layout == "contiguous":
        b = _make_input_storage(K, N, device, input_mode, seed, generator, is_b=True)
    else:
        b_storage = _make_input_storage(N, K, device, input_mode, seed, generator, is_b=True)
        b = b_storage.T
    return a, b
