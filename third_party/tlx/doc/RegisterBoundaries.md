# Register layout and allocation boundaries

TLX exposes explicit boundaries for the two independent compiler decisions
that affect register tensors: their distributed layout and their native AMD
register-allocation interval. These operations preserve numerical values; they
only constrain compilation.

## Layout boundaries

`tlx.require_layout(value, layout)` starts an explicit layout requirement. By
default the requirement is pinned. Passing `pin=False` leaves it available to
layout propagation and conversion folding.

`tlx.release_layout(value)` ends that explicit requirement. The source must
already have a register layout. Consumers after the boundary may select a new
layout, which is useful when a value in a dot-operand layout feeds a reduction
or another operation whose preferred layout differs.

```python
values = tlx.require_layout(values, dot_operand_layout, pin=False)
values = tlx.release_layout(values)
row_sum = tl.sum(values, axis=1)
```

`release_layout` is not a layout conversion request and does not name the
replacement layout. Use another `require_layout` when the consumer layout is a
required part of the algorithm.

## AMD allocation boundaries

`tlx.amd_register_resident(value, register_class, registers_per_group)` makes
all per-thread groups visible to one allocator constraint. Use it when the
complete tensor must remain simultaneously resident across a software-pipeline
phase.

`tlx.amd_register_handoff(value, register_class, registers_per_group)` starts a
new allocation interval for each group independently. It is appropriate for a
local scheduling boundary where whole-tensor simultaneous residency would add
unnecessary register pressure. Each source occurrence is a distinct boundary;
the compiler preserves it across common-subexpression elimination and loop
motion.

Both operations accept `"vgpr"` or `"agpr"`. `registers_per_group` is a
power-of-two count of 32-bit native register values from 1 through 32.
Consequently, one register value contains one 32-bit element or two 16-bit
elements. The tensor's per-thread element count must be divisible by the
resulting native tuple. `amd_register_resident` represents each group as one
allocator-visible register tuple; `amd_register_handoff` constrains the
register values together without promising physical register numbers or
contiguity.

These APIs select a register class and grouping contract, not physical register
numbers. In particular, AGPR placement is not inherently faster: it can reduce
VGPR pressure, but it can also lower occupancy or introduce register-file
moves. Choose it from measured pipeline lifetime and occupancy evidence.
