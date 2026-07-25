"""
Blackwell Symmetric-Memory All-Gather Tutorial
==============================================

Implements a forward-only, single-node All-Gather that reads peer GPU memory
through pointers exported by ``torch.distributed._symmetric_memory``. The
public path selects the best measured B200 strategy:

- payloads below 64 MiB per rank use concurrent fanout peer reads;
- payloads at or above 64 MiB use ring-ordered phased int32 reads;
- odd BF16 shard lengths use phased BF16 reads to preserve slice alignment.

The benchmark compares this adaptive path with
``torch.ops._c10d_functional.all_gather_into_tensor`` followed by
``wait_tensor``. Inputs are already resident in symmetric memory; allocation,
rendezvous, and staging copies are outside the timed region.

Eight-rank B200 median latency in milliseconds:

========  ========  ============  =======
MiB/rank  Strategy  Triton best   c10d-f
========  ========  ============  =======
1         fanout    0.021         0.176
4         fanout    0.061         0.211
16        fanout    0.221         0.259
64        int32     0.844         0.855
128       int32     1.569         1.620
256       int32     2.991         3.150
========  ========  ============  =======

Requirements
------------
- 2, 4, or 8 Blackwell GPUs in one NVLink domain
- NCCL process group
- ``torch.distributed._symmetric_memory``

Examples::

    torchrun --standalone --nproc_per_node=2 \
        third_party/tlx/tutorials/blackwell_dist_all_gather.py --mode correctness

    torchrun --standalone --nproc_per_node=8 \
        third_party/tlx/tutorials/blackwell_dist_all_gather.py --mode benchmark
"""

import argparse
import datetime
import os
import statistics

os.environ.setdefault("TORCH_SYMM_MEM_DISABLE_MULTICAST", "1")

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

DTYPE = torch.bfloat16
BLOCK_SIZE = 4096
MIB = 1024 * 1024
PHASED_THRESHOLD_BYTES = 64 * MIB
DEFAULT_SIZES_MIB = (1, 4, 16, 64, 128, 256)


@triton.jit(do_not_specialize=["local_rank"])
def _fanout_kernel(
    output_ptr,
    buffer_ptrs,
    shard_numel,
    local_rank,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tile = tl.program_id(0)
    peer_step = tl.program_id(1)
    peer_rank = (local_rank + peer_step) % WORLD_SIZE
    peer_addr = tl.load(buffer_ptrs + peer_rank)
    peer_ptr = peer_addr.to(tl.pointer_type(tl.bfloat16))
    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < shard_numel
    values = tl.load(peer_ptr + offsets, mask=mask)
    tl.store(output_ptr + peer_rank * shard_numel + offsets, values, mask=mask)


@triton.jit(do_not_specialize=["phase", "local_rank"])
def _bf16_phase_kernel(
    output_ptr,
    buffer_ptrs,
    shard_numel,
    phase,
    local_rank,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tile = tl.program_id(0)
    peer_rank = (local_rank - phase + WORLD_SIZE) % WORLD_SIZE
    peer_addr = tl.load(buffer_ptrs + peer_rank)
    peer_ptr = peer_addr.to(tl.pointer_type(tl.bfloat16))
    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < shard_numel
    values = tl.load(peer_ptr + offsets, mask=mask)
    tl.store(output_ptr + peer_rank * shard_numel + offsets, values, mask=mask)


@triton.jit(do_not_specialize=["phase", "local_rank"])
def _int32_phase_kernel(
    output_ptr,
    buffer_ptrs,
    shard_numel,
    phase,
    local_rank,
    WORLD_SIZE: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    tile = tl.program_id(0)
    peer_rank = (local_rank - phase + WORLD_SIZE) % WORLD_SIZE
    peer_addr = tl.load(buffer_ptrs + peer_rank)
    peer_words = peer_addr.to(tl.pointer_type(tl.int32))
    output_words = output_ptr.to(tl.pointer_type(tl.int32))
    word_count = shard_numel // 2
    offsets = tile * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
    mask = offsets < word_count
    values = tl.load(peer_words + offsets, mask=mask, other=0)
    output_offset = (peer_rank * shard_numel) // 2
    tl.store(output_words + output_offset + offsets, values, mask=mask)


def allocate_symmetric_input(shape, group=None):
    """Allocate a BF16 symmetric input and expose all peer base pointers."""
    numel = 1
    for dim in shape:
        numel *= dim
    if numel <= 0:
        raise ValueError(f"shape must be non-empty, got {shape}")
    if group is None:
        group = dist.group.WORLD

    raw = symm_mem.empty(
        numel * DTYPE.itemsize,
        dtype=torch.uint8,
        device=torch.cuda.current_device(),
    )
    handle = symm_mem.rendezvous(raw, group=group)
    local_input = handle.get_buffer(rank=handle.rank, sizes=shape, dtype=DTYPE)
    buffer_ptrs = torch.tensor(handle.buffer_ptrs, device=local_input.device, dtype=torch.int64)
    return handle, local_input, buffer_ptrs


def _validate_all_gather_args(output, symmetric_input, handle, buffer_ptrs):
    if symmetric_input.dtype != DTYPE or output.dtype != DTYPE:
        raise TypeError("the tutorial currently supports torch.bfloat16 only")
    if not symmetric_input.is_cuda or not output.is_cuda:
        raise ValueError("input and output must be CUDA tensors")
    if not symmetric_input.is_contiguous() or not output.is_contiguous():
        raise ValueError("input and output must be contiguous")

    world_size = len(handle.buffer_ptrs)
    expected_numel = symmetric_input.numel() * world_size
    if output.numel() != expected_numel:
        raise ValueError(f"output has {output.numel()} elements, expected {expected_numel}")
    if buffer_ptrs.numel() != world_size or buffer_ptrs.dtype != torch.int64:
        raise ValueError("buffer_ptrs must contain one int64 CUDA pointer per rank")
    return world_size


def _strategy_name(symmetric_input):
    payload_bytes = symmetric_input.numel() * symmetric_input.element_size()
    if payload_bytes < PHASED_THRESHOLD_BYTES:
        return "fanout"
    if symmetric_input.numel() % 2 == 0:
        return "int32-phased"
    return "bf16-phased-fallback"


def symm_mem_all_gather_into_tensor(output, symmetric_input, handle, buffer_ptrs):
    """Run the adaptive best measured symmetric-memory All-Gather."""
    world_size = _validate_all_gather_args(output, symmetric_input, handle, buffer_ptrs)
    strategy = _strategy_name(symmetric_input)

    if strategy == "fanout":
        grid = (triton.cdiv(symmetric_input.numel(), BLOCK_SIZE), world_size)
        _fanout_kernel[grid](
            output,
            buffer_ptrs,
            symmetric_input.numel(),
            handle.rank,
            WORLD_SIZE=world_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )
        return output

    if strategy == "int32-phased":
        block_words: tl.constexpr = BLOCK_SIZE // 2
        grid = (triton.cdiv(symmetric_input.numel() // 2, block_words), )
        for phase in range(world_size):
            _int32_phase_kernel[grid](
                output,
                buffer_ptrs,
                symmetric_input.numel(),
                phase,
                handle.rank,
                WORLD_SIZE=world_size,
                BLOCK_WORDS=block_words,
                num_warps=8,
            )
            handle.barrier(channel=phase % 2)
        return output

    grid = (triton.cdiv(symmetric_input.numel(), BLOCK_SIZE), )
    for phase in range(world_size):
        _bf16_phase_kernel[grid](
            output,
            buffer_ptrs,
            symmetric_input.numel(),
            phase,
            handle.rank,
            WORLD_SIZE=world_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )
        handle.barrier(channel=phase % 2)
    return output


def _make_output(symmetric_input, world_size):
    shape = (
        symmetric_input.shape[0] * world_size,
        *symmetric_input.shape[1:],
    )
    return torch.empty(shape, device=symmetric_input.device, dtype=DTYPE)


def _c10d_functional_all_gather(symmetric_input, world_size, group_name):
    pending = torch.ops._c10d_functional.all_gather_into_tensor(symmetric_input, world_size, group_name)
    return torch.ops._c10d_functional.wait_tensor(pending)


def _fill_rank_data(tensor, rank):
    values = torch.arange(tensor.numel(), device=tensor.device, dtype=torch.int64)
    tensor.copy_(((values + rank * 97) % 2048).reshape(tensor.shape))


def check_correctness(shard_numel, handle, symmetric_input, buffer_ptrs):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _fill_rank_data(symmetric_input, rank)
    handle.barrier()

    actual = _make_output(symmetric_input, world_size)
    symm_mem_all_gather_into_tensor(actual, symmetric_input, handle, buffer_ptrs)
    expected = _c10d_functional_all_gather(symmetric_input, world_size, dist.group.WORLD.group_name)
    torch.cuda.synchronize()

    if not torch.equal(actual, expected):
        mismatch = (actual != expected).flatten().nonzero()[0].item()
        raise AssertionError(f"rank={rank}: {_strategy_name(symmetric_input)} mismatch at "
                             f"flat index {mismatch}: triton={actual.flatten()[mismatch].item()}, "
                             f"c10d={expected.flatten()[mismatch].item()}")

    baseline = actual.clone()
    for iteration in range(3):
        symm_mem_all_gather_into_tensor(actual, symmetric_input, handle, buffer_ptrs)
        torch.cuda.synchronize()
        if not torch.equal(actual, baseline):
            raise AssertionError(f"rank={rank}: {_strategy_name(symmetric_input)} "
                                 f"non-deterministic iteration {iteration}")

    handle.barrier()
    if rank == 0:
        print(
            f"correctness PASS: ranks={world_size}, elements/rank={shard_numel}, "
            f"strategy={_strategy_name(symmetric_input)}",
            flush=True,
        )


def _measure_batches(fn, handle, batches, iterations, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    measurements = []
    for _ in range(batches):
        handle.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        local_ms = start.elapsed_time(end) / iterations
        elapsed = torch.tensor(local_ms, device="cuda", dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        measurements.append(elapsed.item())
    return measurements


def _format_spread(samples):
    ordered = sorted(samples)
    p20 = ordered[int(0.2 * (len(ordered) - 1))]
    p80 = ordered[int(0.8 * (len(ordered) - 1))]
    return statistics.median(ordered), p20, p80


def benchmark_size(size_mib, batches, iterations, warmup):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shard_bytes = size_mib * MIB
    shard_numel = shard_bytes // DTYPE.itemsize
    handle, symmetric_input, buffer_ptrs = allocate_symmetric_input((shard_numel, ))
    _fill_rank_data(symmetric_input, rank)
    handle.barrier()

    output = _make_output(symmetric_input, world_size)
    triton_fn = lambda: symm_mem_all_gather_into_tensor(output, symmetric_input, handle, buffer_ptrs)
    functional_fn = lambda: _c10d_functional_all_gather(symmetric_input, world_size, dist.group.WORLD.group_name)

    funcs = [("triton", triton_fn), ("functional", functional_fn)]
    if size_mib.bit_length() % 2:
        funcs.reverse()
    samples = {name: _measure_batches(fn, handle, batches, iterations, warmup) for name, fn in funcs}

    functional_output = functional_fn()
    torch.cuda.synchronize()
    if not torch.equal(output, functional_output):
        raise AssertionError(f"benchmark correctness failed for {size_mib} MiB/rank")

    triton_ms, p20, p80 = _format_spread(samples["triton"])
    functional_ms, _, _ = _format_spread(samples["functional"])
    remote_bytes = shard_bytes * (world_size - 1)
    remote_gbs = remote_bytes / (triton_ms * 1e-3) / 1e9

    handle.barrier()
    if rank == 0:
        print(
            f"{size_mib:9d}  {_strategy_name(symmetric_input):22s}  "
            f"{triton_ms:9.3f} [{p20:.3f},{p80:.3f}]  "
            f"{functional_ms:9.3f}  {triton_ms / functional_ms:7.3f}  "
            f"{remote_gbs:10.1f}",
            flush=True,
        )


def _parse_sizes(value):
    sizes = tuple(int(item) for item in value.split(","))
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive MiB values")
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("correctness", "benchmark"), default="correctness")
    parser.add_argument(
        "--sizes-mib",
        type=_parse_sizes,
        default=DEFAULT_SIZES_MIB,
        help="comma-separated per-rank BF16 payload sizes",
    )
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] < 10:
        raise RuntimeError("this tutorial requires Blackwell GPUs")

    dist.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=datetime.timedelta(minutes=5),
    )
    world_size = dist.get_world_size()
    if world_size not in (2, 4, 8):
        raise ValueError(f"expected 2, 4, or 8 ranks, got {world_size}")

    try:
        if args.mode == "correctness":
            cases = (
                1 * MIB // DTYPE.itemsize,
                PHASED_THRESHOLD_BYTES // DTYPE.itemsize,
                PHASED_THRESHOLD_BYTES // DTYPE.itemsize + 1,
            )
            for shard_numel in cases:
                handle, symmetric_input, buffer_ptrs = allocate_symmetric_input((shard_numel, ))
                check_correctness(shard_numel, handle, symmetric_input, buffer_ptrs)
        else:
            if dist.get_rank() == 0:
                print(f"Blackwell adaptive All-Gather, {world_size} ranks")
                print("Times are median ms [p20,p80], using slowest-rank time.")
                print(
                    " MiB/rank  Strategy                Triton ms [spread]       "
                    "c10d-f ms  T/c10d  Remote GB/s",
                    flush=True,
                )
            for size_mib in args.sizes_mib:
                benchmark_size(size_mib, args.batches, args.iterations, args.warmup)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
