# Harness Template

Prefer a Python harness that can run baseline and TTGIR variants with one entry
point. Keep the launch shape, input construction, and architecture knobs fixed
unless a variant explicitly tests them.

## Per-config `ir_override`

Use this when the kernel already has a `triton.Config` list or can cheaply use
one variant config:

```python
variant_ttgir = os.environ.get("ABLATION_TTGIR")
configs = [
    triton.Config(
        {"BLOCK_SIZE": 32, "ir_override": variant_ttgir}
        if variant_ttgir
        else {"BLOCK_SIZE": 32},
        num_warps=4,
    )
]
```

Run with:

```bash
TRITON_ALWAYS_COMPILE=1 ABLATION_TTGIR=/path/to/variant.ttgir python test_harness.py
```

## Override-directory path

Use this when the dumped directory layout is already available:

```bash
TRITON_ALWAYS_COMPILE=1 \
TRITON_KERNEL_OVERRIDE=1 \
TRITON_OVERRIDE_DIR=/path/to/override_dir \
python test_harness.py
```

The override directory must contain the expected hash subdirectory and matching
TTGIR filename from the dump.

## Oracle-changing assertions

Make expected output depend on the variant semantics, not on the original
kernel, when `oracle_mode` is `replace`.

```python
oracle_mode = os.environ.get("ABLATION_ORACLE", "preserve")

if oracle_mode == "preserve":
    expected = reference_output(inputs)
elif oracle_mode == "replace_constant":
    expected = torch.full_like(output, fill_value=0.0)
elif oracle_mode == "relax":
    expected = None
else:
    raise ValueError(f"unknown oracle mode: {oracle_mode}")

launch_kernel(output, inputs)

if expected is not None:
    torch.testing.assert_close(output, expected)
```

For masked stores, build `expected` with the same mask. For removed stores,
initialize `output` with a sentinel and assert the sentinel remains wherever the
edited IR no longer writes.
