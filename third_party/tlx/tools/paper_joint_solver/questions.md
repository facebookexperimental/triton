# Questions for the authors

Written from an independent reproduction of the paper's Blackwell (B200)
results. We have no H100, so the two Hopper cases are out of scope here and
nothing below is a claim about them.

Each question states what we observed that prompted it, so the answer can be as
narrow as possible. Where we had to choose an interpretation to proceed, we say
which one we chose, so a correction is easy to give.

---

## Priority 1 — these currently block us

### Q1. Could you publish the dependence graphs for the backward pass, as you do for the forward?

Figure 9 gives the discovered warp-specialization strategy for the Blackwell
forward pass as an annotated dependence graph, and it is the only published
*structure* in the paper: the evaluation reports solve times, performance bars
and prose, but no partition for any other case.

For the backward pass we have only the prose of section 6.3.2 -- that the FA4
strategy uses three groups of warps (two reading accumulators from Tensor Memory
to apply exponentials, one staging accumulators in shared memory for the atomic
reduction), that your default run finds a two-group ping-pong strategy, and that
the reduced-register run recovers the three-group one.

That description is not enough to check a partition against. Two Figure-9-style
graphs -- one for each of your two result bars in Figure 11, the default-budget
and the reduced-register runs -- would let us verify the backward the same way
we can verify the forward. If a full figure is
too much, a per-operation group assignment in any machine-readable form would
serve equally well.

### Q2. What are the initiation interval and schedule length of the reported solutions?

We could not find `I` or `L` reported anywhere in the evaluation for any of the
four cases. Since the joint problem's objective is the minimum initiation
interval, these are the natural targets for a reproduction, and without them we
have no numeric quantity to compare against -- only structure and wall-clock
performance.

Could you give `(I, L)`, and the number of pipeline copies, for each reported
result? The forward Blackwell case matters most to us.

### Q3. How is WARPUNIQUENESS extended to operations that span multiple warps?

Figure 6 states the constraint for the single-warp case:

    ∀v  Σ_w opw[v,w] = 1

and the text says it "is naturally extended to operations that span multiple
warps, such as the warp-group level operations on Hopper and Blackwell", but the
extended formula is not given. Section 4 describes the output as "an assignment
of every v ∈ V to a warp (or warps)".

Two things are underdetermined for us, and they interact:

1. **The cardinality.** We read the extension as `Σ_w opw[v,w] = min_warps(v)`,
   i.e. a warp-group operation occupies exactly four warps. Is that your
   reading, and where does `min_warps` come from in your pipeline?

2. **What makes two operations "the same warp".** This is the one that bites.
   If the relation is *equality of warp sets*, then a group containing both a
   single-warp operation and a four-warp operation is unsatisfiable: the shared
   set would need cardinality 1 and 4 at once. The structure Figure 9 describes
   appears to have exactly this shape -- softmax groups that contain both
   warp-group-level and single-warp operations -- so under the equality reading
   we cannot represent it at all, and any solver pinned to it returns UNSAT
   regardless of costs, `I` or `L`.

   We proceeded by reading "same group" as *containment* (a one-warp operation
   belongs to the group whose four-warp set contains its warp), which makes the
   structure expressible. Is that what your implementation does, or does `opw`
   range over warp *groups* rather than physical warps, making the question moot?

---

## Priority 2 — we had to choose an interpretation and would like to check it

### Q4. Does the CONCURRENCY window apply in full to asynchronous producers?

Figure 6 excludes issue points in `[t − (cycles(o) − 1), t]`, with no case split
on the producer's pipeline. Read literally, an operation whose `cycles` is large
excludes a correspondingly large window on any warp it shares.

That appears to be in tension with the mechanism you describe in section 3.2 and
Figure 2: tensor-core and TMA operations run on their functional unit while the
issuing warp proceeds. Concretely, a schedule that issues a Tensor Memory load a
few cycles after the MMA it waits on, in the same group, violates the literal
window when `cycles(MMA)` is large -- yet overlapping those two is a large part
of what the pipeline is for.

Does your implementation apply the full `cycles(o)` window to TC/TMA producers,
or a shorter one (for example, the issue cycle only)? We assumed the latter.

### Q5. What are the members of the cost list C, and what does the budget sum over?

Section 5.2 says "let the list of integers C be the original cycle counts" and
bounds it with `1 ≤ Σ_i C'[i] ≤ U`, with `U = 300`.

Two things we could not settle from the text:

1. **Membership.** Which quantities join C -- per-operation latencies,
   reservation-table durations, edge latencies, spill costs, and are zeros
   included? We build C from reservation-table maximal-segment durations, edge
   latencies and per-node spill costs.

2. **The index set of the sum.** Does `Σ_i C'[i]` run over every *occurrence* in
   C, or over its *distinct values*? On our inputs the two readings differ by
   roughly a factor of 2.5 in the resolution the same `U` buys, which is large
   enough to change the solutions we get.

Relatedly: was `U = 300` used for all four reported cases, or does it vary per
case?

### Q6. How do compiler cycle measurements project onto the reservation table?

The paper defines the final reservation table but, as far as we can tell, not
the projection from measured per-operation cycle counts onto it. We split each
table into maximal constant-demand runs and normalize the run durations
independently, which keeps the integer demand vectors exact, but this is our
construction rather than yours.

Could you describe the projection you use? This sits upstream of everything
else, so a difference here moves every downstream number.

---

## Priority 3 — useful for matching your numbers, not blocking

### Q7. What register-per-thread budget produced the reduced-register backward run?

Section 6.3.2 says you re-ran with "a reduced register-per-thread budget" after
ptxas spilled on the two-group schedule. What value did you use? We are
currently guessing at the reduction, and since the whole point of that run is
that the budget changes which strategy is optimal, the specific value matters.

### Q8. What pipeline depths were used for the streaming operations?

Section 5.3 exposes the pipeline depth of a streaming operation "as a parameter
to an external auto-tuning system". Since a streaming producer's outgoing
latency is zeroed, the depth is what makes that treatment sound on hardware.
What depths did the reported results use, and were they auto-tuned or chosen by
hand?

### Q9. Is the register-limit model reconciled with ptxas anywhere?

The backward story -- the model admits a two-group schedule, ptxas then spills
on it, and tightening the budget recovers a schedule ptxas can allocate -- reads
like a known gap between REGISTERLIMIT and the real allocator. Do you have a
rule of thumb for the margin, or is tightening the budget until ptxas succeeds
the intended workflow?

---

## What we are not asking about

We are not asking for code or for the hand-compiled CUDA. Structure and
parameters are enough for us to check our reproduction against yours; the
questions above are all answerable in a paragraph or a small file each.
