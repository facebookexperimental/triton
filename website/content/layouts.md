> On how TLX defers layout computation until after function inlining — needed
> when a helper's effective `num_warps` differs from the module's — see
> [Placeholder Layouts in TLX](third_party/tlx/doc/PlaceholderLayouts.md).

- `tlx.dump_layout(x)` **[sm90+, gfx942+]**

    Compile-time diagnostic that prints the resolved layout of a value to the
    compiler log. `x` may be a register tensor or a shared/tensor-memory buffer
    (memdesc). It emits **no** device code and returns nothing — this is a
    static, host-side diagnostic, distinct from the runtime `tl.device_print` /
    `tl.print`. The op is rendered at the end of the TTGIR pipeline, so the
    printed layout reflects all compiler optimizations, then it is erased.

    The layout is printed in CuTe (CUTLASS) `Shape:Stride` notation (`_N` marks
    a static integer):
    - Register tensors → a thread-value (TV) layout
      `((thread...),(value...)):((thread...),(value...))`, where the thread
      group comes from the hardware lane/warp/block dims and the value group
      from the per-thread registers (stride `_0` denotes a broadcast).
    - Shared/tensor-memory buffers → a single strided layout, e.g. `_64:_1`.
    - Swizzled shared buffers → `Swizzle<B,M,S> o (base):(stride)`.

    In all cases the layout maps a coordinate to the **logical tensor's
    row-major element index** (its codomain): for a register tensor a
    `(thread, value)` coordinate → the logical element index it holds, and for
    a buffer an offset → the buffer element index. The strides are offsets in
    that logical index space, not physical byte/bank addresses.

    Layouts that are not representable as a CuTe layout fall back to the raw
    linear-layout string.

    Example:
    ```python
    x = tl.load(x_ptr + offs)          # register tensor
    tlx.dump_layout(x)                  # -> // cute: ((_32,_2,_2),_1):((_1,_32,_0),_0)

    buf = tlx.local_alloc((BLOCK,), tl.float32, 1)
    v = tlx.local_view(buf, 0)
    tlx.dump_layout(v)                  # -> // cute: _64:_1
    ```

- `x = tlx.require_layout(x, layout, pin=True, late_address_compute=False)` **[sm90+, gfx942+]**

    Require a register tensor `x` to use `layout`, expressed as a
    `tlx.layout(...)` (Shape:Stride). With the default `pin=True`, the `#linear`
    encoding is wrapped as `#tlx.no_verify_layout(#tlx.user_layout(...))`. The
    inner `#tlx.user_layout` carries `PinnedEncodingTrait`, so downstream passes
    (`tritongpu-coalesce`, `remove-layout-conversions`, AMD `optimize-epilogue`)
    treat it as fixed and never rewrite it; the outer `#tlx.no_verify_layout` defers
    operand-layout verification until the pin is peeled by
    `TLXResolvePlaceholderLayouts` in `make_ttgir`. Example: pin an FP16 epilogue
    `tl.store` to a coalesced layout so `OptimizeEpilogue` keeps the wide
    `buffer_store_dwordx4` instead of narrowing it to the MMA-accumulator store,
    without staging the value through LDS.

    Pass `pin=False` for an optimizer-flexible requirement. This emits the
    requested encoding without the `#tlx.user_layout` hard anchor, allowing
    later layout passes to propagate the requirement or materialize a layout
    conversion. The default remains `pin=True` for existing callers.

    On AMD, `late_address_compute=True` asks shared-memory-backed layout
    conversions to compute their addresses at this use. The backend does this
    by rematerializing inexpensive lane/warp coordinates, shortening their live
    ranges across register-heavy regions.

    Pair with `tlx.assert_same_layout(x, layout)` (below) to statically verify the
    pin survived to the final TTGIR.

- `x = tlx.release_layout(x)` **[sm90+, gfx942+]**

    End an explicit register-layout requirement without changing the tensor's
    value. Downstream layout propagation may select a layout preferred by the
    consumer, for example when a dot-operand layout feeds a reduction. This is
    not a conversion request; use another `require_layout` when the replacement
    layout is part of the algorithm.

- `tlx.assert_same_layout(lhs, rhs)` **[sm90+, gfx942+]**

    Compile-time assertion that two layouts are equivalent after layout
    propagation and all other TTGIR layout optimizations have completed. Like
    `tlx.dump_layout`, it emits no device code and is consumed at the end of the
    TTGIR pipeline.

    `rhs` supports two forms:

    - **Value/value:** `rhs` is another register tensor or shared/tensor-memory
      buffer. The frontend emits `tlx.assert_same_layout`, whose two operands
      retain their independently resolved final types.
    - **Value/layout:** `rhs` is a constant `tlx.layout_encoding`. The frontend
      lowers the constant to an encoding attribute and emits
      `tlx.assert_same_layout_expected`. At assertion time, the pass combines
      that encoding with `lhs`'s shape, element type, and (for buffers) memory
      properties to construct an expected tensor or memdesc type.

    These are separate internal operations only because an SSA value is an MLIR
    operand while a constant layout is an MLIR attribute. They share the same
    comparison path and the same public Python API.

    Before comparison, both final types are converted with
    `ttg::toLinearLayout`. The assertion compares the resulting
    `LinearLayout`s, not the original encoding attributes. Consequently,
    structurally different encodings pass if they describe the same logical
    mapping. A mismatch reports both normalized LinearLayouts and fails
    compilation.

    Example:

    ```python
    x = tlx.local_load(x_buf, layout=REGISTER_LAYOUT)
    y = tlx.local_load(y_buf, layout=REGISTER_LAYOUT)

    tlx.assert_same_layout(x, y)                # value/value
    tlx.assert_same_layout(x, REGISTER_LAYOUT)  # value/layout
    ```

## Explicit WMMA layout pinning

`tlx.require_amd_wmma_layout(x, version=3, transposed=True, warp_bases=...,
reg_bases=..., instr_shape=(16, 16, 128))` pins a tensor to an explicit AMD
WMMA register/warp layout. This is useful for tuned gfx1250 epilogues that must
retain the accumulator ownership chosen by `tiles_per_warp` across otherwise
layout-neutral tensor operations. The bases contain one linear-layout basis
vector per register or warp bit and must match the tensor rank.

The helper lowers to a pinned `tlx.require_layout`; omit it when automatic
layout propagation is sufficient.


## Compiler pipeline inspection

To introspect the pipeline `add_stages`, before running your kernels, simply set
the add_stages_inspection_hook like so:

```python
def inspect_stages(_self, stages, options, language, capability):
    # inspect or modify add_stages here
triton.knobs.runtime.add_stages_inspection_hook = inspect_stages
```

For an example of using this with out-of-tree plugin passes, see
[`examples/plugins/`](examples/plugins/README.md).
