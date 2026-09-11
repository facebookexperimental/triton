"""L1 correctness for the gfx950 KDA public operators."""

from itertools import accumulate, pairwise

import pytest
import torch
import triton
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import kda_paged_prefill, kda_recurrent_decode

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="gfx950 KDA operators require CDNA4")

DEVICE = triton.runtime.driver.active.get_active_torch_device()
ARCH = "gfx950"


def _kda_recurrent_reference(q, k, v, g, beta, state, scale):
    """FP32 recurrence for prepared inputs and a physical [H, V, K] state."""
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.to(torch.bfloat16).float()
    beta = beta.float()

    state = state.float().clone()
    outputs = []
    for token in range(q.shape[0]):
        state *= g[token].exp()[:, None, :]
        prediction = torch.einsum("hvk,hk->hv", state, k[token])
        delta = beta[token, :, None] * (v[token] - prediction)
        state += torch.einsum("hv,hk->hvk", delta, k[token])
        outputs.append(scale * torch.einsum("hvk,hk->hv", state, q[token]))
    if not outputs:
        return v.new_empty((0, *v.shape[1:]), dtype=torch.float32), state
    return torch.stack(outputs), state


def _normalized_kda_input(shape):
    value = torch.randn(shape, device=DEVICE, dtype=torch.float32)
    return torch.nn.functional.normalize(value, dim=-1).to(torch.bfloat16)


def _check_kda_prefill(lengths, heads, scale, cu_on_cpu=False):
    torch.manual_seed(31)
    dim = 128
    total_tokens = sum(lengths)
    shape = (1, total_tokens, heads, dim)
    q = _normalized_kda_input(shape)
    k = _normalized_kda_input(shape)
    v = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    g = -torch.nn.functional.softplus(torch.randn(shape, device=DEVICE, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(1, total_tokens, heads, device=DEVICE, dtype=torch.float32))
    initial_state = 0.1 * torch.randn(
        len(lengths),
        heads,
        dim,
        dim,
        device=DEVICE,
        dtype=torch.float32,
    )
    cu_device = "cpu" if cu_on_cpu else DEVICE
    cu_seqlens = torch.tensor([0, *accumulate(lengths)], device=cu_device, dtype=torch.int64)

    expected_outputs = []
    expected_states = []
    boundaries = cu_seqlens.tolist()
    for sequence, (begin, end) in enumerate(pairwise(boundaries)):
        expected_output, expected_state = _kda_recurrent_reference(
            q[0, begin:end],
            k[0, begin:end],
            v[0, begin:end],
            g[0, begin:end],
            beta[0, begin:end],
            initial_state[sequence],
            scale,
        )
        expected_outputs.append(expected_output)
        expected_states.append(expected_state)

    actual_output, actual_state = kda_paged_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        arch=ARCH,
    )
    torch.testing.assert_close(
        actual_output[0].float(),
        torch.cat(expected_outputs),
        atol=8e-3,
        rtol=3e-2,
    )
    torch.testing.assert_close(
        actual_state,
        torch.stack(expected_states),
        atol=8e-3,
        rtol=3e-2,
    )
    assert actual_state.dtype == torch.float32
    assert actual_state.shape == (len(lengths), heads, dim, dim)
    assert actual_state.stride()[-2:] == (dim, 1)


@pytest.mark.parametrize("scale", [1.0, 128**-0.5], ids=["unit-scale", "attention-scale"])
def test_kda_paged_prefill_chunk_boundaries(scale):
    _check_kda_prefill(
        [0, 1, 15, 16, 17, 63, 64, 65],
        heads=12,
        scale=scale,
        cu_on_cpu=True,
    )


@pytest.mark.parametrize(
    ("lengths", "heads"),
    [
        pytest.param([64], 12, id="single-chunk-h12"),
        pytest.param([256, 255, 128, 17], 4, id="ragged-multichunk-h4"),
    ],
)
def test_kda_paged_prefill_smoke_shapes(lengths, heads):
    _check_kda_prefill(lengths, heads=heads, scale=128**-0.5)


def _make_kda_decode_inputs(batch, heads, key_dim, value_dim, strided):
    q_shape = (1, batch, heads, key_dim)
    if strided:
        packed = torch.empty(
            1,
            batch,
            heads * (2 * key_dim + value_dim) + 7,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        q_end = heads * key_dim
        k_end = 2 * q_end
        v_end = k_end + heads * value_dim
        q = packed[..., :q_end].view(q_shape)
        k = packed[..., q_end:k_end].view(q_shape)
        v = packed[..., k_end:v_end].view(1, batch, heads, value_dim)
        q.copy_(_normalized_kda_input(q_shape))
        k.copy_(_normalized_kda_input(q_shape))
        v.normal_()
        gate_storage = torch.empty(1, batch, heads * key_dim + 5, device=DEVICE, dtype=torch.float32)
        g = gate_storage[..., :q_end].view(q_shape)
        g.copy_(-torch.nn.functional.softplus(torch.randn_like(g)))
        beta_storage = torch.empty(1, batch, heads + 3, device=DEVICE, dtype=torch.float32)
        beta = beta_storage[..., :heads]
        beta.copy_(torch.sigmoid(torch.randn_like(beta)))
    else:
        q = _normalized_kda_input(q_shape)
        k = _normalized_kda_input(q_shape)
        v = torch.randn(1, batch, heads, value_dim, device=DEVICE, dtype=torch.bfloat16)
        g = -torch.nn.functional.softplus(torch.randn(q_shape, device=DEVICE, dtype=torch.float32))
        beta = torch.sigmoid(torch.randn(1, batch, heads, device=DEVICE, dtype=torch.float32))
    return q, k, v, g, beta


@pytest.mark.parametrize(
    ("heads", "key_dim", "value_dim", "strided_inputs", "scale_mode"),
    [
        pytest.param(2, 1, 1, False, "unit", id="minimal"),
        pytest.param(2, 3, 5, True, "attention", id="masked-strided"),
        pytest.param(2, 8, 5, False, "unit", id="mixed-dim"),
        pytest.param(12, 128, 128, True, "attention", id="production"),
    ],
)
def test_kda_recurrent_decode_indexed_state(heads, key_dim, value_dim, strided_inputs, scale_mode):
    torch.manual_seed(13)
    batch = 3
    scale = key_dim**-0.5 if scale_mode == "attention" else 1.0
    q, k, v, g, beta = _make_kda_decode_inputs(batch, heads, key_dim, value_dim, strided_inputs)
    state_pool = torch.randn(7, heads, value_dim, key_dim, device=DEVICE, dtype=torch.float32)
    original_pool = state_pool.clone()
    expected_pool = state_pool.clone()
    read_indices = torch.tensor([0, 1, 1], device=DEVICE, dtype=torch.int32)
    write_indices = torch.tensor([2, 3, 4], device=DEVICE, dtype=torch.int32)
    cu_seqlens = torch.arange(batch + 1, device=DEVICE, dtype=torch.int64)

    expected_output = []
    for row in range(batch):
        output, final_state = _kda_recurrent_reference(
            q[0, row : row + 1],
            k[0, row : row + 1],
            v[0, row : row + 1],
            g[0, row : row + 1],
            beta[0, row : row + 1],
            original_pool[read_indices[row].long()],
            scale,
        )
        expected_output.append(output[0])
        expected_pool[write_indices[row].long()] = final_state
    expected_output = torch.stack(expected_output).unsqueeze(0)
    actual_output = kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        arch=ARCH,
    )

    torch.testing.assert_close(actual_output.float(), expected_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state_pool, expected_pool, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(state_pool[0], original_pool[0], atol=0, rtol=0)
    torch.testing.assert_close(state_pool[1], original_pool[1], atol=0, rtol=0)


def test_kda_recurrent_decode_graph_padding_and_slot_stride():
    torch.manual_seed(23)
    batch, active, heads, key_dim, value_dim = 4, 2, 2, 8, 5
    state_elements = heads * key_dim * value_dim
    raw_pool = torch.randn(7, state_elements + 11, device=DEVICE, dtype=torch.float32)
    padding_before = raw_pool[:, state_elements:].clone()
    state_pool = raw_pool[:, :state_elements].view(7, heads, value_dim, key_dim)
    original_pool = state_pool.clone()
    q, k, v, g, beta = _make_kda_decode_inputs(batch, heads, key_dim, value_dim, strided=True)
    read_indices = torch.tensor([1, 2, -1, -1], device=DEVICE, dtype=torch.int32)
    write_indices = torch.tensor([3, 4, -1, -1], device=DEVICE, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1, 2, 2, 2], device=DEVICE, dtype=torch.int32)
    kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        arch=ARCH,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kda_recurrent_decode(
            q,
            k,
            v,
            g,
            beta,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=cu_seqlens,
            arch=ARCH,
        )
    graph.replay()
    torch.cuda.synchronize()

    expected_pool = original_pool.clone()
    expected_output = []
    for row in range(active):
        row_output, final_state = _kda_recurrent_reference(
            q[0, row : row + 1],
            k[0, row : row + 1],
            v[0, row : row + 1],
            g[0, row : row + 1],
            beta[0, row : row + 1],
            original_pool[read_indices[row].long()],
            1.0,
        )
        expected_output.append(row_output[0])
        expected_pool[write_indices[row].long()] = final_state
    torch.testing.assert_close(
        captured[:, :active].float(),
        torch.stack(expected_output).unsqueeze(0),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(captured[:, active:], torch.zeros_like(captured[:, active:]))
    torch.testing.assert_close(state_pool, expected_pool, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(raw_pool[:, state_elements:], padding_before, atol=0, rtol=0)
    for untouched in (0, 1, 2, 5, 6):
        torch.testing.assert_close(state_pool[untouched], original_pool[untouched], atol=0, rtol=0)


@pytest.mark.parametrize(
    ("invalid_index", "scale"),
    [
        pytest.param(-1, 1.0, id="negative-unit-scale"),
        pytest.param(7, 8**-0.5, id="out-of-range-attention-scale"),
    ],
)
def test_kda_recurrent_decode_invalid_indices(scale, invalid_index):
    torch.manual_seed(41)
    batch, heads, key_dim, value_dim = 2, 2, 8, 5
    q, k, v, g, beta = _make_kda_decode_inputs(batch, heads, key_dim, value_dim, strided=False)
    pool = torch.randn(7, heads, value_dim, key_dim, device=DEVICE, dtype=torch.float32)
    original_pool = pool.clone()
    reads = torch.tensor([invalid_index, 0], device=DEVICE, dtype=torch.int32)
    writes = torch.tensor([5, invalid_index], device=DEVICE, dtype=torch.int32)
    cu_seqlens = torch.arange(batch + 1, device=DEVICE, dtype=torch.int32)

    expected_output = [torch.zeros_like(v[0, 0], dtype=torch.float32)]
    output1, _ = _kda_recurrent_reference(
        q[0, 1:2],
        k[0, 1:2],
        v[0, 1:2],
        g[0, 1:2],
        beta[0, 1:2],
        original_pool[0],
        scale,
    )
    expected_output.append(output1[0])
    actual = kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        state_pool=pool,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=cu_seqlens,
        arch=ARCH,
    )
    torch.testing.assert_close(
        actual.float(),
        torch.stack(expected_output).unsqueeze(0),
        atol=2e-2,
        rtol=2e-2,
    )
    for untouched in range(pool.shape[0]):
        torch.testing.assert_close(pool[untouched], original_pool[untouched], atol=0, rtol=0)


def test_kda_recurrent_decode_cpu_metadata_and_malformed_rows():
    torch.manual_seed(53)
    batch, heads, key_dim, value_dim = 3, 2, 8, 5
    q, k, v, g, beta = _make_kda_decode_inputs(batch, heads, key_dim, value_dim, strided=False)
    pool = torch.randn(6, heads, value_dim, key_dim, device=DEVICE, dtype=torch.float32)
    original_pool = pool.clone()
    expected_pool = pool.clone()
    reads = torch.tensor([0, 1, 2], dtype=torch.int32)
    writes = torch.tensor([3, 4, 5], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 3, 3], dtype=torch.int64)

    expected_output, expected_state = _kda_recurrent_reference(
        q[0, 2:3],
        k[0, 2:3],
        v[0, 2:3],
        g[0, 2:3],
        beta[0, 2:3],
        original_pool[1],
        1.0,
    )
    expected_pool[4] = expected_state
    actual = kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        state_pool=pool,
        read_indices=reads,
        write_indices=writes,
        cu_seqlens=cu_seqlens,
        arch=ARCH,
    )

    torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]))
    torch.testing.assert_close(actual[:, 1].float(), expected_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[:, 2], torch.zeros_like(actual[:, 2]))
    torch.testing.assert_close(pool, expected_pool, atol=2e-4, rtol=2e-4)
