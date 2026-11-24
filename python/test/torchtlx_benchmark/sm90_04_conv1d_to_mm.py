# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""(K, 1) convolution forward and backward at production shapes.

Baseline: cuDNN / ATen convolution with torchTLX off. Candidate: torchTLX allowed, so
tlx_config.conv1d_to_mm lowers the conv to GEMMs and fused pointwise ops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NAME = "sm90_04_conv1d_to_mm"
CANDIDATE_NAME = "conv1d_to_mm"
_COMMON_CONFIG: dict[str, object] = {
    "force_disable_caches": True,
    "enable_caching_generated_triton_templates": False,
}
BASELINE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": None}
CANDIDATE_CONFIG: dict[str, object] = {**_COMMON_CONFIG, "triton.tlx_mode": "allow"}
REL_L2_TOL = 1.0e-2
# x, weight
GRAD_INPUTS = (0, 1)

# Input length is the output length plus the K - 1 rows the graph pre-pads (padding=0).
SHAPES = (
    {"batch": 1365, "channels": 96, "length": 2502, "kernel": 3, "padding": 0, "groups": 1, "channels_last": True},
    {"batch": 1365, "channels": 96, "length": 1922, "kernel": 3, "padding": 0, "groups": 1, "channels_last": True},
    {"batch": 1365, "channels": 96, "length": 2500, "kernel": 1, "padding": 0, "groups": 1, "channels_last": False},
    {"batch": 1365, "channels": 96, "length": 2500, "kernel": 3, "padding": 1, "groups": 96, "channels_last": False},
    {"batch": 1365, "channels": 96, "length": 202, "kernel": 3, "padding": 0, "groups": 1, "channels_last": True},
    {"batch": 1365, "channels": 96, "length": 302, "kernel": 3, "padding": 0, "groups": 1, "channels_last": True},
    {"batch": 1365, "channels": 256, "length": 100, "kernel": 3, "padding": 1, "groups": 1, "channels_last": True},
)
N, C, LENGTH, K, PADDING, GROUPS, CHANNELS_LAST = SHAPES[0].values()


def add_arguments(parser) -> None:
    parser.add_argument("--batch", type=int)
    parser.add_argument("--channels", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--kernel", type=int)
    parser.add_argument("--padding", type=int)
    parser.add_argument("--groups", type=int)
    parser.add_argument("--channels-last", dest="channels_last", type=int)


def configure(args) -> None:
    global N, C, LENGTH, K, PADDING, GROUPS, CHANNELS_LAST
    N = args.batch or N
    C = args.channels or C
    LENGTH = args.length or LENGTH
    K = args.kernel or K
    PADDING = PADDING if args.padding is None else args.padding
    GROUPS = args.groups or GROUPS
    CHANNELS_LAST = CHANNELS_LAST if args.channels_last is None else bool(args.channels_last)


def problem() -> str:
    layout = "NHWC" if CHANNELS_LAST else "NCHW"
    return f"N={N} C={C} L={LENGTH} K={K} pad={PADDING} groups={GROUPS} {layout} dtype=bf16"


def model(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    import torch

    return torch.nn.functional.conv2d(x, weight, padding=(PADDING, 0), groups=GROUPS)


def make_inputs() -> tuple[torch.Tensor, ...]:
    import torch

    torch.manual_seed(0)
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    x = torch.randn((N, C, LENGTH, 1), **bf16)
    weight = torch.randn((C, C // GROUPS, K, 1), **bf16) / (C // GROUPS * K)**0.5
    if CHANNELS_LAST:
        x = x.to(memory_format=torch.channels_last)
        weight = weight.to(memory_format=torch.channels_last)
    return x, weight
