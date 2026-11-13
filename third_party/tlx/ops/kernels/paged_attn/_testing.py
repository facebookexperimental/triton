"""Independent FP64 reference and fixtures for page64 BF16 decode tests/benchmarks.

Not imported by the operator. Reference attention gathers only visible keys,
so unused page-table entries and cache slots can be poisoned independently.
"""

import torch
import triton


def make_inputs(lengths, query_length, *, seed=41, device="cuda"):
    batch = len(lengths)
    pages = max(1, triton.cdiv(max(lengths), 64))
    generator = torch.Generator(device=device).manual_seed(seed)
    query = torch.randn((batch * query_length, 16, 64), generator=generator, device=device, dtype=torch.bfloat16)
    key = torch.randn((batch * pages + 3, 2, 64, 64), generator=generator, device=device, dtype=query.dtype)
    value = torch.randn(key.shape, generator=generator, device=device, dtype=query.dtype)
    table = torch.randperm(key.shape[0], generator=generator, device=device, dtype=torch.int32)
    table = table[:batch * pages].reshape(batch, pages).contiguous()
    return query, key, value, torch.tensor(lengths, dtype=torch.int32, device=device), table


def pack_cache(key, value):
    """Standard [P,H,T,D] to K[P,H,D/8,T,8], V[P,H,T/8,D,8]."""
    pages = key.shape[0]
    key = key.reshape(pages, 2, 64, 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    value = value.reshape(pages, 2, 8, 8, 64).permute(0, 1, 2, 4, 3).contiguous()
    return key, value


def reference(query, key, value, lengths, table, query_length, window=0, scale=0.125):
    """Compute every output element in FP64, independently of the split plan."""
    result = torch.zeros_like(query, dtype=torch.float64)
    for batch, length in enumerate(lengths.tolist()):
        if length == 0:
            continue
        first = max(0, length - query_length - window + 1) if window else 0
        positions = torch.arange(first, length, device=query.device)
        physical = table[batch, positions // 64].long()
        keys = key[physical, :, positions % 64].double().permute(1, 0, 2)
        values = value[physical, :, positions % 64].double().permute(1, 0, 2)
        rows = query[batch * query_length:(batch + 1) * query_length]
        queries = rows.double().reshape(query_length, 2, 8, 64).permute(1, 0, 2, 3).reshape(2, query_length * 8, 64)
        qpos = length - query_length + torch.arange(query_length * 8, device=query.device) // 8
        mask = positions[None, :] <= qpos[:, None]
        if window:
            mask &= positions[None, :] > qpos[:, None] - window
        scores = (queries @ keys.transpose(-1, -2)) * scale
        probabilities = scores.masked_fill(~mask, -float("inf")).softmax(-1).nan_to_num(0.)
        output = (probabilities @ values).reshape(2, query_length, 8, 64).permute(1, 0, 2, 3)
        result[batch * query_length:(batch + 1) * query_length] = output.reshape(query_length, 16, 64)
    return result


def assert_close(actual, expected):
    actual = actual.double()
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
    rrms = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
    assert rrms < 0.005, f"FP64 relative RMS {rrms.item():.6f} exceeds 0.005"
    return rrms.item()
