# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch reference for ``tlx.ops.hstu_cross_attn``.

A dense PyTorch implementation over padded sequences, using plain torch for the
jagged/dense conversions.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F


def _get_valid_attn_mask(
    device: torch.device,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    seq_lengths_q: torch.Tensor,
    seq_lengths_kv: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    targets_in_kv: bool = True,
) -> torch.Tensor:
    col_ids = torch.arange(0, max_seq_len_q, device=device).view(1, max_seq_len_q, 1)
    row_ids = torch.arange(0, max_seq_len_kv, device=device).view(1, 1, max_seq_len_kv)
    in_boundary_valid_attn_mask = torch.logical_and(row_ids < seq_lengths_kv.view(-1, 1, 1), col_ids
                                                    < seq_lengths_q.view(-1, 1, 1))
    if not causal:
        return in_boundary_valid_attn_mask
    if num_targets is None:
        delta_col_ids = seq_lengths_kv - seq_lengths_q
        shifted_col_ids = col_ids + delta_col_ids.view(-1, 1, 1)
        return torch.logical_and(in_boundary_valid_attn_mask, shifted_col_ids >= row_ids)
    uih_lengths_q = seq_lengths_q - num_targets
    delta_col_ids = seq_lengths_kv - uih_lengths_q
    if targets_in_kv:
        delta_col_ids = delta_col_ids - num_targets
    shifted_col_ids = col_ids + delta_col_ids.view(-1, 1, 1)
    causal_mask = torch.logical_and(col_ids < uih_lengths_q.view(-1, 1, 1), shifted_col_ids >= row_ids)
    target_mask = col_ids >= uih_lengths_q.view(-1, 1, 1)
    if targets_in_kv:
        target_mask = torch.logical_and(
            target_mask,
            torch.logical_or(row_ids < (seq_lengths_kv - num_targets).view(-1, 1, 1), shifted_col_ids == row_ids),
        )
    return torch.logical_and(in_boundary_valid_attn_mask, torch.logical_or(causal_mask, target_mask))


def _to_padded(x: torch.Tensor, offsets: List[int], max_len: int) -> torch.Tensor:
    """Jagged ``(total, ...)`` to padded ``(B, max_len, ...)``."""
    out = x.new_zeros((len(offsets) - 1, max_len, *x.shape[1:]))
    for b in range(len(offsets) - 1):
        start, end = offsets[b], offsets[b + 1]
        out[b, :end - start] = x[start:end]
    return out


def _to_jagged(x: torch.Tensor, offsets: List[int]) -> torch.Tensor:
    """Padded ``(B, max_len, ...)`` to jagged ``(total, ...)``."""
    return torch.cat([x[b, :offsets[b + 1] - offsets[b]] for b in range(len(offsets) - 1)], dim=0)


def pytorch_hstu_cross_mha(
    max_seq_len_q: int,
    max_seq_len_kv: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    seq_offsets_kv: torch.Tensor,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    targets_in_kv: bool = False,
) -> torch.Tensor:
    """Jagged cross attention: ``q`` is ``(total_q, H, D)``, ``k``/``v`` ``(total_kv, H_kv, D/V)``.

    The first ``num_softmax_heads`` heads use softmax, the rest SiLU scaled by
    ``attn_scale``. GQA repeats KV heads to match the query heads.
    """
    total_q, H, _ = q.shape
    H_kv = k.shape[1]
    assert num_softmax_heads <= H
    offs_q = seq_offsets_q.tolist()
    offs_kv = seq_offsets_kv.tolist()
    pq = _to_padded(q, offs_q, max_seq_len_q).transpose(1, 2)  # [B, H, Nq, D]
    pk = _to_padded(k, offs_kv, max_seq_len_kv).transpose(1, 2)  # [B, H_kv, Nkv, D]
    pv = _to_padded(v, offs_kv, max_seq_len_kv).transpose(1, 2)  # [B, H_kv, Nkv, V]
    if H > H_kv:
        assert H % H_kv == 0, f"H ({H}) must be divisible by H_kv ({H_kv})"
        pk = pk.repeat_interleave(H // H_kv, dim=1)
        pv = pv.repeat_interleave(H // H_kv, dim=1)
    qk_attn = torch.einsum("bhxa,bhya->bhxy", pq.float(), pk.float()) * alpha
    qk_attn_softmax = qk_attn[:, :num_softmax_heads]
    qk_attn_silu = qk_attn[:, num_softmax_heads:]
    if attn_scale.ndim > 0:
        attn_scale = _to_padded(attn_scale.unsqueeze(-1), offs_q, max_seq_len_q).unsqueeze(1).to(q.dtype)
    valid_attn_mask = _get_valid_attn_mask(
        device=q.device,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_kv=max_seq_len_kv,
        seq_lengths_q=seq_offsets_q[1:] - seq_offsets_q[:-1],
        seq_lengths_kv=seq_offsets_kv[1:] - seq_offsets_kv[:-1],
        num_targets=num_targets,
        causal=causal,
        targets_in_kv=targets_in_kv,
    ).unsqueeze(1)
    qk_max = torch.max(qk_attn_softmax, dim=-1, keepdim=True)[0]
    qk_attn_softmax = torch.where(valid_attn_mask, qk_attn_softmax - qk_max, -float("inf"))
    qk_attn_softmax = torch.exp(qk_attn_softmax)
    qk_attn_softmax = qk_attn_softmax / (torch.sum(qk_attn_softmax, dim=-1, keepdim=True) + 1e-6)
    qk_attn_softmax = torch.where(valid_attn_mask, qk_attn_softmax, 0.0).to(q.dtype)
    qk_attn_silu = F.silu(qk_attn_silu.to(q.dtype)) * attn_scale
    qk_attn_silu = torch.where(valid_attn_mask, qk_attn_silu, 0.0)
    qk_attn = torch.cat((qk_attn_softmax, qk_attn_silu), dim=1)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, pv)  # [B, H, Nq, V]
    return _to_jagged(attn_dense.transpose(1, 2), offs_q)  # [total_q, H, V]
