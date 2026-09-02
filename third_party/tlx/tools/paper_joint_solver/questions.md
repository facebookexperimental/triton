# Questions for the authors

### Q1. Could you publish the dependence graphs (input and solution) for the backward pass, as Figure 9 does for the forward?

For the backward pass we have only the prose of section 6.3.2: that the FA4
strategy uses three groups of warps (two reading accumulators from Tensor Memory
to apply exponentials, one staging accumulators in shared memory for the atomic
reduction), that your default run finds a two-group ping-pong strategy, and that
the reduced-register run recovers the three-group one.

Concretely: a Figure-9-style graph for each of the two result bars in
Figure 11 (the default-budget and the reduced-register runs), or a
per-operation group assignment in any machine-readable form.

We would also value the *input* dependence graphs (the nodes, edges, and the
cycle counts fed to the normalization) for any of the four cases, in any
format.

### Q2. What are the initiation interval and schedule length of the reported solutions?

### Q3. How is WARPUNIQUENESS extended to operations that span multiple warps?

Figure 6 states the constraint for the single-warp case:

    ∀v  Σ_w opw[v,w] = 1

and the text says it "is naturally extended to operations that span multiple
warps, such as the warp-group level operations on Hopper and Blackwell", but the
extended formula is not given.

### Q4. What makes the solver prefer solutions with few groups?

### Q5. Does the CONCURRENCY window apply in full to asynchronous producers?

Figure 6 excludes issue points in `[t − (cycles(o) − 1), t]`, with no case split
on the producer's pipeline. Read literally, an operation whose `cycles` is large
excludes a correspondingly large window on any warp it shares.

That appears to be in tension with the mechanism you describe in section 3.2 and
Figure 2: tensor-core and TMA operations run on their functional unit while the
issuing warp proceeds. Concretely, a schedule that issues a Tensor Memory load a
few cycles after the MMA it waits on, in the same group, violates the literal
window when `cycles(MMA)` is large, yet overlapping those two is a large part
of what the pipeline is for. In our reproduction, the strategy shaped like your
published forward one contains exactly such pairs, so under the literal window
it is rejected.

Does your implementation apply the full `cycles(o)` window to TC/TMA producers,
or a shorter one (for example, the issue cycle only)?

### Q6. How do per-operation cycle estimates project onto the reservation table?

The paper defines the final reservation table but, as far as we can tell, not
the projection onto it from the per-operation cycle estimates (which section 5
says come from documentation or direct measurement).

### Q7. Does Algorithm 1 really ascend from I = 1, and what dominates the reported solve times?

Algorithm 1 as printed starts at `I ← 0` and increments, with no lower bound.
Run literally on our inputs, total solve time is dominated by the unsatisfiable
prefix (the modulo-scheduling calls and joint attempts at every I below the
answer), and single hard instances can run for hours; our reduced-register
backward case did not terminate a single joint attempt in 24 hours under one
input preparation.

### Q8. What register-per-thread budget produced the reduced-register backward run?

Section 6.3.2 says you re-ran with "a reduced register-per-thread budget" after
ptxas spilled on the two-group schedule. What value did you use?

### Q9. What pipeline depths were used for the streaming operations, and what exactly is zeroed?

Section 5.3 exposes the pipeline depth of a streaming operation "as a parameter
to an external auto-tuning system". What depths did the reported results use?

### Q10. What is the machine's warp budget, and how much did the warp-reduction ablation reduce?

Section 6.2.2 lists "reducing the number of warps" among the modifications that
make the problem unsatisfiable, without saying from what to what.
