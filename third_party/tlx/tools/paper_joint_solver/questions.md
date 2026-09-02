# Questions for the authors

Written from an independent reproduction of the paper's Blackwell (B200)
results. We have no H100, so the two Hopper cases are out of scope here and
nothing below is a claim about them.

Where we stand, for context: our transcription of the constraint system
(Figures 4/5/6, Algorithm 1, the section 5.2 normalization) reproduces your
three UNSAT ablations, and — under input-preparation assumptions we disclose in
our tree — the initiation-interval regime and solve-time regime of your
forward and backward results. What we cannot yet reproduce without further
interpretation is the *shape* of the reported solutions (the five-group
forward strategy, the two-group backward ping-pong). Most questions below come
from that gap.

Each question states what we observed that prompted it, so the answer can be as
narrow as possible. Where we had to choose an interpretation to proceed, we say
which one we chose, so a correction is easy to give. Appendix A lists every
interpretation we adopted in one table.

---

## Priority 1 — these currently block us

### Q1. Could you publish the dependence graphs — input and solution — for the backward pass, as Figure 9 does for the forward?

Figure 9 gives the discovered warp-specialization strategy for the Blackwell
forward pass as an annotated dependence graph, and it is the only published
*structure* in the paper: the evaluation reports solve times, performance bars
and prose, but no partition for any other case.

For the backward pass we have only the prose of section 6.3.2 — that the FA4
strategy uses three groups of warps (two reading accumulators from Tensor Memory
to apply exponentials, one staging accumulators in shared memory for the atomic
reduction), that your default run finds a two-group ping-pong strategy, and that
the reduced-register run recovers the three-group one.

That description is not enough to check a partition against. Two Figure-9-style
graphs — one for each of your two result bars in Figure 11, the default-budget
and the reduced-register runs — would let us verify the backward the same way
we can verify the forward. If a full figure is too much, a per-operation group
assignment in any machine-readable form would serve equally well.

The same request one level up would settle even more: the *input* dependence
graphs (nodes, edges, and the cycle counts you fed the normalization) for any
of the four cases, in any format. Almost every question in this document is a
question about quantities that would simply be visible in one such file.

### Q2. What are the initiation interval and schedule length of the reported solutions?

We could not find `I` or `L` reported anywhere in the evaluation for any of the
four cases. Since the joint problem's objective is the minimum initiation
interval, these are the natural targets for a reproduction, and without them we
have no numeric quantity to compare against — only structure and wall-clock
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
   appears to have exactly this shape — softmax groups that contain both
   warp-group-level and single-warp operations — so under the equality reading
   we cannot represent it at all, and any solver pinned to it returns UNSAT
   regardless of costs, `I` or `L`.

   We proceeded by reading "same group" as *containment* (a one-warp operation
   belongs to the group whose four-warp set contains its warp), which makes the
   structure expressible — and, in our reproduction, makes pinning the
   five-group forward partition satisfiable where it was structurally UNSAT
   before. Is containment what your implementation does, or does `opw` range
   over warp *groups* rather than physical warps, making the question moot?

### Q4. What makes the solver prefer solutions with few groups?

This is distinct from Q3: Q3 is about whether the published shape is
*expressible*; this is about why it is *selected*.

Nothing in Figure 6 couples one operation's warp assignment to another's beyond
the variable-latency pinning, and nothing rewards reusing warps. Under our
transcription the SMT solver behaves accordingly: free solves scatter
operations across 17–18 distinct warp sets on a 32-warp machine, where the
reported strategies use 2–5 groups. The scattered assignments satisfy every
published constraint, so as far as the published model is concerned they are
equally optimal.

We measured the gap from the other side as well: constraining the solve to at
most G distinct warp sets, the smallest satisfiable G at our forward optimum is
4 — so few-group solutions exist in the optimal set; the model just never
prefers them.

Does your implementation carry something the paper does not state — a bounded
pool of groups that `opw` ranges over, a secondary objective, a symmetry
reduction with this side effect, or a post-processing step — that yields the
small group counts in the reported results?

---

## Priority 2 — we had to choose an interpretation and would like to check it

### Q5. Does the CONCURRENCY window apply in full to asynchronous producers?

Figure 6 excludes issue points in `[t − (cycles(o) − 1), t]`, with no case split
on the producer's pipeline. Read literally, an operation whose `cycles` is large
excludes a correspondingly large window on any warp it shares.

That appears to be in tension with the mechanism you describe in section 3.2 and
Figure 2: tensor-core and TMA operations run on their functional unit while the
issuing warp proceeds. Concretely, a schedule that issues a Tensor Memory load a
few cycles after the MMA it waits on, in the same group, violates the literal
window when `cycles(MMA)` is large — yet overlapping those two is a large part
of what the pipeline is for. In our reproduction, the strategy shaped like your
published forward one contains exactly such pairs, so under the literal window
it is rejected.

Does your implementation apply the full `cycles(o)` window to TC/TMA producers,
or a shorter one (for example, the issue cycle only)? We assumed the latter.

### Q6. What are the members of the cost list C, and what does the budget sum over?

Section 5.2 says "let the list of integers C be the original cycle counts" and
bounds it with `1 ≤ Σ_i C'[i] ≤ U`, with `U = 300`.

Three things we could not settle from the text:

1. **Membership.** Which quantities join C — per-operation latencies,
   reservation-table durations, edge latencies, spill costs, and are zeros
   included? We build C from reservation-table maximal-segment durations, edge
   latencies and per-node spill costs.

2. **The index set of the sum.** Does `Σ_i C'[i]` run over every *occurrence* in
   C, or over its *distinct values*? On our inputs the two readings differ by
   roughly a factor of 2.5 in the resolution the same `U` buys, which is large
   enough to change the solutions we get. (A related observation, in case it is
   useful: on our per-instruction lists, the minimum-F optimum at `U = 300` is
   a near-total collapse — most entries priced at zero — because the objective
   is an absolute cross-product error and zeroing small entries is free. The
   distinct-values reading avoids this on our inputs. If your C was small or
   deduplicated, the phenomenon would simply never have arisen for you.)

3. **The tie-break.** The F-optimal face generally contains many points,
   degenerate and rich ones alike, and the paper does not say which one is
   returned. Did your implementation add a secondary criterion, or did the
   solver's default choice happen to be usable?

Relatedly: we read "In our experiments, we pick U = 300" as one global value
across all four reported cases — is that right? And — smallest possible ask
with the largest value — could you share the concrete `C` and `C'` for the
Blackwell forward case?

### Q7. How do per-operation cycle estimates project onto the reservation table?

The paper defines the final reservation table but, as far as we can tell, not
the projection onto it from the per-operation cycle estimates (which section 5
says come from documentation or direct measurement). We split each
table into maximal constant-demand runs and normalize the run durations
independently, which keeps the integer demand vectors exact, but this is our
construction rather than yours.

Could you describe the projection you use? This sits upstream of everything
else, so a difference here moves every downstream number.

### Q8. Does Algorithm 1 really ascend from I = 1, and what dominates the reported solve times?

Algorithm 1 as printed starts at `I ← 0` and increments, with no lower bound.
Run literally on our inputs, total solve time is dominated by the unsatisfiable
prefix — the modulo-scheduling calls and joint attempts at every I below the
answer — and single hard instances can run for hours; our reduced-register
backward case did not terminate a single joint attempt in 24 hours under one
input preparation, and solved in seconds under another.

For the reported 19 s / 269 s / 64 s: does your implementation seed the ascent
at a resource or recurrence lower bound (MinII) rather than 1, warm-start the
SMT queries, or bound them in time? Section 5.2 already accounts for the
normalization ZLP (global minima in under 500 ms in all cases), so the split
we cannot see is between the modulo-scheduling ILP calls (CBC) and the SMT
calls (Yices): roughly how does the reported time divide between those two?
This would tell us whether a large timing discrepancy on our side signals a
wrong input world or merely a different search prefix.

---

## Priority 3 — useful for matching your numbers, not blocking

### Q9. What register-per-thread budget produced the reduced-register backward run?

Section 6.3.2 says you re-ran with "a reduced register-per-thread budget" after
ptxas spilled on the two-group schedule. What value did you use? We are
currently guessing at the reduction (we run 128 regs/thread against a full 255),
and since the whole point of that run is that the budget changes which strategy
is optimal, the specific value matters.

### Q10. What pipeline depths were used for the streaming operations, and what exactly is zeroed?

Section 5.3 exposes the pipeline depth of a streaming operation "as a parameter
to an external auto-tuning system". Since a streaming producer's outgoing
latency is zeroed, the depth is what makes that treatment sound on hardware.
What depths did the reported results use, and were they auto-tuned or chosen by
hand?

One scope question underneath: "we assign streaming operations zero latency in
our cost models" can mean the operation's own execution span (its reservation
table becomes empty) or only its outgoing dependence latencies (it still
occupies its functional unit). The two differ materially in the CAPACITY
constraint. We zero only the outgoing latencies; which did you intend?

### Q11. Is the register-limit model reconciled with ptxas anywhere?

The backward story — the model admits a two-group schedule, ptxas then spills
on it, and tightening the budget recovers a schedule ptxas can allocate — reads
like a known gap between REGISTERLIMIT and the real allocator. Do you have a
rule of thumb for the margin, or is tightening the budget until ptxas succeeds
the intended workflow?

### Q12. What is the machine's warp budget, and how much did the warp-reduction ablation reduce?

Section 6.2.2 lists "reducing the number of warps" among the modifications that
make the problem unsatisfiable, without saying from what to what. We model the
machine at W = 32 physical warps and probe two cuts, W−1 = 31 and W/2 = 16.
What W does your machine description use, and what reduced value produced the
reported UNSAT?

---

## What we are not asking about

We are not asking for code or for the hand-compiled CUDA. Structure and
parameters are enough for us to check our reproduction against yours; the
questions above are all answerable in a paragraph or a small file each.

---

## Appendix A — every interpretation we adopted, in one table

These are disclosed in our tree (`DEVIATIONS.md`, "The prepared regime"); the
table maps each to the question whose answer would confirm or retire it. Items
marked *paper* are the paper's own content, listed for completeness.

| # | Interpretation | Hangs on |
|---|---|---|
| A1 | Cost pool: one entry per node, plus non-zero edge latencies and spills | Q6.1 |
| A2 | Elementwise ops re-priced at `ceil(tile elements / 128)`; scalar/address arithmetic at zero | Q7 |
| A3 | Asynchronous MMA results treated as tokens, not scalars | Q7 |
| A4 | Tensor-Memory ports modeled as their own functional unit | Q7 |
| B1 | 10% relative-tolerance clustering of measured costs before the ZLP | Q6.1 / Q7 |
| B2 | Entries below `max/32` zeroed before the ZLP | Q6.1 |
| B3 | The section 5.2 ZLP itself — *paper*, imported unmodified | — |
| B4 | Lexicographic second stage: maximize `Σ C'` at `F = F*` | Q6.3 |
| B5 | Per-case U — **retired**; U = 300 everywhere once B7 is adopted | Q6 |
| B6 | F-relaxation ladder `(1,2,3,4,6,8)`, first rung clearing our health gate | Q6.3 |
| B7 | The budget sums over *distinct values* of C, not occurrences | Q6.2 |
| 11 | "Same group" = containment of warp sets, not equality | Q3.2 |
| 12 | CONCURRENCY window for TC/TMA producers = issue cycle only | Q5 |
| — | Warp-group cardinality `Σ_w opw = min_warps(v)` | Q3.1 |
| — | Reduced register budget = 128 regs/thread | Q9 |
| — | Machine warp budget W = 32; ablation cuts 31 and 16 | Q12 |
| — | Streaming zero-latency = outgoing edges only | Q10 |
| — | UNSAT ablations verified by first-window enumeration (Algorithm 1 alone cannot terminate on UNSAT) | Q8 |

## Appendix B — measurements still open on our side

Answers to the questions above would land into an evaluation with these items
still in flight; none of them changes what the questions ask.

- The minimal group counts for the backward cases are known only as intervals
  (default budget: > 4; reduced budget: ≤ 6) — the counting-constrained solves
  are our most expensive instances. Q1's group assignments would replace the
  intervals outright.
- The direction check "reduced register budget uses more groups" is therefore
  unresolved on our side; it awaits either the intervals closing or Q1/Q9.
- Our corrected-semantics batch (assumptions 11/12 enabled end to end) is in
  progress; the shape verdicts quoted in this document are from measurements at
  pinned schedule points, not yet from free solves under the corrected
  semantics.
