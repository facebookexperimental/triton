"""tlx.ops.hstu_attn correctness -- gfx950."""
from pathlib import Path
import pytest
import torch
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops.kernels.hstu_attn._shapes import CORRECTNESS_SHAPES, inputs

GFX950_SHAPES = [
    (2, 128, 2, 128, 128),
    (4, 256, 4, 128, 128),
]

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("Z,MAX_SEQ_LEN,H,HEAD_DIM,causal,dtype_name", CORRECTNESS_SHAPES)
def test_hstu_attn_common_shapes_gfx950(Z, MAX_SEQ_LEN, H, HEAD_DIM, causal, dtype_name):
    from triton.tlx.ops import hstu_attn_dev as tlx_hstu_attn
    from triton.tlx.ops.kernels.hstu_attn._reference import triton_hstu_mha

    dtype = DTYPES[dtype_name]
    q, k, v, offsets, attn_scale = inputs(Z, MAX_SEQ_LEN, H, HEAD_DIM, dtype)
    alpha = 1.0 / HEAD_DIM
    actual = tlx_hstu_attn(q, k, v, offsets, MAX_SEQ_LEN, attn_scale, alpha=alpha, causal=causal, space="smoke")
    expected = triton_hstu_mha(MAX_SEQ_LEN, alpha, q, k, v, offsets, attn_scale)
    precision = 1e-3 if dtype == torch.float16 else 8e-3
    torch.testing.assert_close(actual, expected, atol=precision * expected.abs().max().item(), rtol=precision)


def _gfx950_inputs(batch_size, max_seq_len, H, attn_dim, hidden_dim, dtype):
    device = torch.device("cuda")
    lengths = torch.linspace(max_seq_len // 2, max_seq_len, batch_size, device=device, dtype=torch.int32)
    offsets = torch.zeros((batch_size + 1, ), dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lengths.to(torch.int64), dim=0)
    total = int(offsets[-1].item())
    x = torch.empty((total, H, attn_dim * 2 + hidden_dim), dtype=dtype, device=device).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)
    num_targets = torch.clamp(lengths // 4, min=1)
    return q.contiguous(), k.contiguous(), v.contiguous(), offsets, num_targets


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize("batch_size, MAX_SEQ_LEN, H, ATTN_DIM, HIDDEN_DIM", GFX950_SHAPES)
def test_hstu_attn_gfx950(batch_size, MAX_SEQ_LEN, H, ATTN_DIM, HIDDEN_DIM):
    from triton.tlx.ops import hstu_attn_dev as tlx_hstu_attn
    from triton.tlx.ops.kernels.hstu_attn.gfx950 import torch_hstu_attention as torch_hstu_attn_ref

    torch.cuda.empty_cache()
    dtype = torch.bfloat16
    alpha = 10000.0 / ATTN_DIM
    q, k, v, offsets, num_targets = _gfx950_inputs(batch_size, MAX_SEQ_LEN, H, ATTN_DIM, HIDDEN_DIM, dtype)

    out = tlx_hstu_attn(q, k, v, offsets, MAX_SEQ_LEN, None, alpha=alpha, causal=True, num_targets=num_targets,
                        space="smoke")
    ref = torch_hstu_attn_ref(
        MAX_SEQ_LEN,
        alpha,
        q,
        k,
        v,
        offsets,
        causal=True,
        dropout_pr=0.0,
        training=False,
        num_targets=num_targets,
    )

    torch.testing.assert_close(out * MAX_SEQ_LEN, ref * MAX_SEQ_LEN, atol=1e-3, rtol=0)


def _load_gfx950_hstu_tutorial():
    """Load the standalone gfx950 HSTU forward/backward implementation."""
    import sys

    kernel_dir = str(Path(__file__).resolve().parents[4] / "third_party" / "tlx" / "tutorials" / "hstu_self_attn")
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    import tlx_gfx950_ragged_hstu_attention as hstu

    return hstu


def _load_gfx950_hstu_benchmark():
    """Load the benchmark helpers without requiring a GPU workload."""
    _load_gfx950_hstu_tutorial()
    import bench_gfx950_bwd as benchmark

    return benchmark


def test_hstu_attn_gfx950_fixture_launch_coverage(tmp_path):
    benchmark = _load_gfx950_hstu_benchmark()

    # Issue #2005 intentionally has a length one row beyond N, but every
    # forward configuration still launches through row 1023.
    assert benchmark._validate_fixture_launch_coverage(996, torch.tensor([898, 997])) == 1024
    assert benchmark._validate_fixture_launch_coverage(1024, torch.tensor([1024])) == 1024

    with pytest.raises(ValueError, match=r"sequence 1 has length 1153.*N=1024.*at most 1024"):
        benchmark._validate_fixture_launch_coverage(1024, torch.tensor([1024, 1153]))

    invalid_fixture = {
        "N": 1024,
        "alpha": 1.0 / 128,
        "q_shape": (2177, 4, 128),
        "seq_offsets": torch.tensor([0, 1024, 2177]),
        "invalid_attn_mask_type": "lower_triangular",
        "num_targets": torch.tensor([20, 20]),
        "attn_bias": None,
        "seq2_offsets": None,
        "max_attn_len": 0,
        "contextual_seq_len": 0,
        "sort_by_length": False,
    }
    fixture_path = tmp_path / "invalid_hstu_fixture.pt"
    torch.save(invalid_fixture, fixture_path)
    with pytest.raises(ValueError, match=r"sequence 1 has length 1153.*N=1024.*at most 1024"):
        benchmark._make_input_fixture_workload(fixture_path)


def test_hstu_attn_gfx950_sequence_xcd_padding_budget():
    hstu = _load_gfx950_hstu_tutorial()

    assert hstu._gfx950_fa_schedule_launch_sequences(7, 8) == (7, False)
    assert hstu._gfx950_fa_schedule_launch_sequences(9, 8) == (9, False)
    assert hstu._gfx950_fa_schedule_launch_sequences(13, 8) == (13, False)
    assert hstu._gfx950_fa_schedule_launch_sequences(14, 8) == (16, True)
    assert hstu._gfx950_fa_schedule_launch_sequences(512, 8) == (512, True)
    assert hstu._gfx950_fa_schedule_launch_sequences(513, 8) == (520, True)
    for num_xcds in (2, 4, 8):
        assert hstu._gfx950_fa_schedule_launch_sequences(15, num_xcds) == (16, True)


def _target_causal_hstu_ref(q, k, v, offsets, num_targets, max_seq_len, alpha):
    """Float reference for history-causal plus independent-target masking."""
    qf = q.float().detach().requires_grad_()
    kf = k.float().detach().requires_grad_()
    vf = v.float().detach().requires_grad_()
    outputs = []
    for z in range(offsets.numel() - 1):
        start, end = int(offsets[z]), int(offsets[z + 1])
        seq_len = end - start
        history_end = seq_len - int(num_targets[z])
        query = torch.arange(seq_len, device=q.device)[:, None]
        key = torch.arange(seq_len, device=q.device)[None, :]
        valid = (query == key) | ((key < history_end) & (key < query))
        scores = torch.einsum("qhd,khd->hqk", qf[start:end], kf[start:end]) * alpha
        weights = scores * torch.sigmoid(scores) / max_seq_len
        outputs.append(torch.einsum("hqk,khd->qhd", weights * valid[None], vf[start:end]))
    return torch.cat(outputs), qf, kf, vf


def _relative_l2(got, expected):
    return ((got.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)).item()


def _assert_per_sequence_close(name, got, expected, offsets, tolerance=8e-3, tail_tolerance=2e-2):
    """Keep short ragged segments and their last valid rows visible in the error."""
    for z in range(offsets.numel() - 1):
        start, end = int(offsets[z]), int(offsets[z + 1])
        error = _relative_l2(got[start:end], expected[start:end])
        assert error < tolerance, f"{name}[{z}] relative L2 {error:.3e} >= {tolerance:.3e}"
        tail_error = _relative_l2(got[end - 1], expected[end - 1])
        assert tail_error < tail_tolerance, f"{name}[{z}] last-row relative L2 {tail_error:.3e} >= {tail_tolerance:.3e}"


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize(
    "bwd_variant,sequence_xcd_case",
    [
        pytest.param("default", "padded", id="default"),
        pytest.param("kv_parallel_fa_schedule", "padded", id="fa-schedule"),
        pytest.param(
            "kv_parallel_fa_schedule_mask_peel_resident_k_dr_early_do_t",
            "padded",
            id="fa-schedule-production",
        ),
        pytest.param(
            "kv_parallel_fa_schedule_bn256_direct_qdo_g2l",
            "padded",
            id="fa-schedule-bn256",
        ),
        pytest.param(
            "kv_parallel_fa_schedule_mask_peel_resident_k_dr_early_do_t",
            "unpadded",
            id="fa-schedule-production-unpadded-sequence-xcd",
        ),
    ],
)
def test_hstu_attn_gfx950_backward_target_causal(bwd_variant, sequence_xcd_case):
    """Cover ragged tails and padded/unpadded XCD sequence scheduling."""
    hstu = _load_gfx950_hstu_tutorial()
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    max_seq_len, heads, head_dim = 512, 2, 128
    alpha = 1.0 / head_dim**0.5
    # The padded cases use fifteen sequences, which pad to sixteen on 2-, 4-,
    # and 8-XCD partitions while staying within the dummy-sequence budget.
    # Exercise a wholly invalid second Q/dO pipeline slot, both sides of the
    # 64-, 128-, and 256-row boundaries, partial final K/V tiles, and two
    # exact BN256 tiles.
    lengths_list = [1, 17, 33, 63, 65, 97, 127, 129, 193, 255, 256, 257, 321, 511, 512]
    if sequence_xcd_case == "unpadded":
        # A multiple of every supported gfx950 XCD count selects sequence-XCD
        # scheduling without compiling the padded-slot guard.
        lengths_list.insert(-2, 400)
    lengths = torch.tensor(lengths_list, device=device, dtype=torch.int64)
    offsets = torch.zeros(lengths.numel() + 1, device=device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    # Include a diagonal-only all-target sequence and history/target boundaries
    # that fall inside 64-, 128-, and 256-row tiles.
    num_targets_list = [1, 1, 5, 1, 65, 17, 20, 33, 5, 20, 17, 20, 17, 5, 20]
    if sequence_xcd_case == "unpadded":
        num_targets_list.insert(-2, 20)
        cu_count = torch.cuda.get_device_properties(device).multi_processor_count
        num_xcds = max(1, min(hstu.GFX950_MAX_XCDS, cu_count // hstu.GFX950_CUS_PER_XCD))
        scheduled_z, use_sequence_xcd = hstu._gfx950_fa_schedule_launch_sequences(len(lengths_list), num_xcds)
        assert use_sequence_xcd
        assert scheduled_z == len(lengths_list)
    num_targets = torch.tensor(num_targets_list, device=device, dtype=torch.int32)
    total = int(offsets[-1])

    q, k, v = (torch.empty((total, heads, head_dim), device=device, dtype=dtype).uniform_(-2.0, 2.0).requires_grad_()
               for _ in range(3))
    dout = torch.empty_like(q).uniform_(-1.0, 1.0)
    ref, q_ref, k_ref, v_ref = _target_causal_hstu_ref(
        q,
        k,
        v,
        offsets,
        num_targets,
        max_seq_len,
        alpha,
    )
    out = hstu.tlx_gfx950_hstu_mha(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        offsets,
        num_targets=num_targets,
        bwd_variant=bwd_variant,
    )
    out.backward(dout)
    ref.backward(dout.float())

    _assert_per_sequence_close("out", out, ref, offsets)
    for name, got, expected in (
        ("dq", q.grad, q_ref.grad),
        ("dk", k.grad, k_ref.grad),
        ("dv", v.grad, v_ref.grad),
    ):
        assert got is not None and expected is not None, name
        _assert_per_sequence_close(name, got, expected, offsets)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
def test_hstu_attn_gfx950_backward_ticket_2005_layout():
    """Cover the interleaved QKV layout and N/length boundary from issue #2005."""
    hstu = _load_gfx950_hstu_tutorial()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    max_seq_len, heads, head_dim = 996, 4, 128
    alpha = 1.0 / head_dim**0.5
    lengths = torch.tensor([898, 997], device=device, dtype=torch.int64)
    offsets = torch.zeros(lengths.numel() + 1, device=device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    num_targets = lengths
    total = int(offsets[-1])

    value_gen = torch.Generator(device=device).manual_seed(1)
    backing = torch.empty((total, 4 * heads, head_dim), device=device,
                          dtype=dtype).uniform_(-2.0, 2.0, generator=value_gen)
    q, k, v, _ = torch.split(backing, [heads, heads, heads, heads], dim=1)
    q, k, v = (tensor.detach().requires_grad_() for tensor in (q, k, v))
    expected_stride = (4 * heads * head_dim, head_dim, 1)
    assert q.stride() == k.stride() == v.stride() == expected_stride
    dout_gen = torch.Generator(device=device).manual_seed(2)
    dout = torch.empty_like(q).uniform_(-1.0, 1.0, generator=dout_gen)
    assert dout.stride() == (heads * head_dim, head_dim, 1)

    ref, q_ref, k_ref, v_ref = _target_causal_hstu_ref(
        q,
        k,
        v,
        offsets,
        num_targets,
        max_seq_len,
        alpha,
    )
    out = hstu.tlx_gfx950_hstu_mha(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        offsets,
        num_targets=num_targets,
        bwd_variant="kv_parallel_fa_schedule_mask_peel_resident_k_dr_early_do_t",
    )
    out.backward(dout)
    ref.backward(dout.float())

    _assert_per_sequence_close("out", out, ref, offsets)
    for name, got, expected in (
        ("dq", q.grad, q_ref.grad),
        ("dk", k.grad, k_ref.grad),
        ("dv", v.grad, v_ref.grad),
    ):
        assert got is not None and expected is not None, name
        _assert_per_sequence_close(name, got, expected, offsets)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize(
    "bwd_options",
    [
        pytest.param(
            {
                "native_mfma_dq_warps": 0,
                "kv_parallel": True,
            },
            id="aliased-dq-acc",
        ),
        pytest.param(
            {
                "native_mfma_dq_warps": 4,
                "kv_parallel": True,
            },
            id="separate-dq-acc",
        ),
        pytest.param(
            {
                "native_mfma_dq_warps": 4,
                "kv_parallel": True,
                "fa_schedule": True,
                "fa_schedule_direct_qdo_g2l": True,
                "fa_schedule_mask_peel": True,
                "fa_schedule_resident_k_score": True,
                "fa_schedule_dr_resident": True,
                "fa_schedule_early_do_t": True,
            },
            id="separate-dq-acc-fa-schedule",
        ),
    ],
)
def test_hstu_attn_gfx950_backward_reuses_poisoned_outputs(bwd_options):
    """Repeated launches must clear aliased or separate dQ accumulation state."""
    hstu = _load_gfx950_hstu_tutorial()
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    max_seq_len, heads, head_dim = 256, 2, 128
    alpha = 1.0 / head_dim**0.5
    lengths = torch.tensor([65, 129, 193], device=device, dtype=torch.int64)
    offsets = torch.zeros(lengths.numel() + 1, device=device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    num_targets = torch.tensor([65, 20, 33], device=device, dtype=torch.int32)
    total = int(offsets[-1])

    q, k, v = (torch.empty((total, heads, head_dim), device=device, dtype=dtype).uniform_(-2.0, 2.0).requires_grad_()
               for _ in range(3))
    dout = torch.empty_like(q).uniform_(-1.0, 1.0)
    ref, q_ref, k_ref, v_ref = _target_causal_hstu_ref(
        q,
        k,
        v,
        offsets,
        num_targets,
        max_seq_len,
        alpha,
    )
    ref.backward(dout.float())
    expected = (q_ref.grad, k_ref.grad, v_ref.grad)
    assert all(tensor is not None for tensor in expected)

    dq, dk, dv = (torch.empty_like(q) for _ in range(3))
    first_result = None
    for poison in (float("nan"), 16384.0, -16384.0):
        for tensor in (dq, dk, dv):
            tensor.fill_(poison)
        hstu.tlx_gfx950_ragged_attention_bwd(
            dout=dout,
            q=q,
            k=k,
            v=v,
            dq=dq,
            dk=dk,
            dv=dv,
            seq_offsets=offsets,
            num_targets=num_targets,
            attn_scale=None,
            N=max_seq_len,
            alpha=alpha,
            max_attn_len=0,
            invalid_attn_mask_type="lower_triangular",
            contextual_seq_len=0,
            sort_by_length_indices=None,
            full_attn_size=0,
            **bwd_options,
        )

        result = (dq, dk, dv)
        for name, got, wanted in zip(("dq", "dk", "dv"), result, expected):
            assert torch.isfinite(got).all(), f"{name} retained poison {poison}"
            _assert_per_sequence_close(name, got, wanted, offsets)
        if first_result is None:
            first_result = tuple(tensor.clone() for tensor in result)
        else:
            for name, got, first in zip(("dq", "dk", "dv"), result, first_result):
                _assert_per_sequence_close(f"repeat-{name}", got, first, offsets, tolerance=2e-3)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware (CDNA4)")
@pytest.mark.parametrize(
    "bwd_options",
    [
        pytest.param(
            {
                "native_mfma_dq_warps": 0,
                "kv_parallel": True,
            },
            id="aliased-dq-acc",
        ),
        pytest.param(
            {
                "native_mfma_dq_warps": 4,
                "kv_parallel": True,
                "fa_schedule": True,
            },
            id="separate-dq-acc-fa-schedule",
        ),
    ],
)
def test_hstu_attn_gfx950_backward_graph_replay_resets_dq(bwd_options):
    """Captured backward launches must reset dQ accumulation on every replay."""
    hstu = _load_gfx950_hstu_tutorial()
    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    max_seq_len, heads, head_dim = 256, 2, 128
    alpha = 1.0 / head_dim**0.5
    lengths = torch.tensor([65, 193], device=device, dtype=torch.int64)
    offsets = torch.zeros(lengths.numel() + 1, device=device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    num_targets = torch.tensor([20, 33], device=device, dtype=torch.int32)
    total = int(offsets[-1])
    q, k, v = (torch.empty((total, heads, head_dim), device=device, dtype=dtype).uniform_(-2.0, 2.0) for _ in range(3))
    dout = torch.empty_like(q).uniform_(-1.0, 1.0)
    dq, dk, dv = (torch.empty_like(q) for _ in range(3))

    def launch():
        hstu.tlx_gfx950_ragged_attention_bwd(
            dout=dout,
            q=q,
            k=k,
            v=v,
            dq=dq,
            dk=dk,
            dv=dv,
            seq_offsets=offsets,
            num_targets=num_targets,
            attn_scale=None,
            N=max_seq_len,
            alpha=alpha,
            max_attn_len=0,
            invalid_attn_mask_type="lower_triangular",
            contextual_seq_len=0,
            sort_by_length_indices=None,
            full_attn_size=0,
            **bwd_options,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    expected_dq = dq.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for _ in range(3):
        dq.fill_(float("nan"))
        graph.replay()
        assert torch.isfinite(dq).all()
        _assert_per_sequence_close("dq", dq, expected_dq, offsets, tolerance=2e-3)
