# Questions held back for a later round

## Q6. What are the members of the cost list C, and what does the budget sum over?

Section 5.2 says "let the list of integers C be the original cycle counts" and
bounds it with `1 ≤ Σ_i C'[i] ≤ U`, with `U = 300`.

Three things we could not settle from the text:

1. **Membership.** Which quantities join C (per-operation latencies,
   reservation-table durations, edge latencies, spill costs), and are zeros
   included? We build C from reservation-table maximal-segment durations, edge
   latencies and per-node spill costs.

2. **The index set of the sum.** Does `Σ_i C'[i]` run over every *occurrence* in
   C, or over its *distinct values*? On our inputs the two readings differ by
   roughly a factor of 2.5 in the resolution the same `U` buys, which is large
   enough to change the solutions we get. (A related observation, in case it is
   useful: on our per-instruction lists, the minimum-F optimum at `U = 300` is
   a near-total collapse (most entries priced at zero) because the objective
   is an absolute cross-product error and zeroing small entries is free. The
   distinct-values reading avoids this on our inputs. If your C was small or
   deduplicated, the phenomenon would simply never have arisen for you.)

3. **The tie-break.** The F-optimal face generally contains many points,
   degenerate and rich ones alike, and the paper does not say which one is
   returned. Did your implementation add a secondary criterion, or did the
   solver's default choice happen to be usable?

Relatedly: we read "In our experiments, we pick U = 300" as one global value
across all four reported cases; is that right? And one small ask with a large
value: could you share the concrete `C` and `C'` for the Blackwell forward
case?

## Q11. Is the register-limit model reconciled with ptxas anywhere?

The backward story (the model admits a two-group schedule, ptxas then spills
on it, and tightening the budget recovers a schedule ptxas can allocate) reads
like a known gap between REGISTERLIMIT and the real allocator. Do you have a
rule of thumb for the margin, or is tightening the budget until ptxas succeeds
the intended workflow?
