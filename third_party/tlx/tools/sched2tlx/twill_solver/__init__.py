"""twill_solver — Twill-inspired joint software-pipelining + warp-specialization
optimal solver for the sched2tlx toolchain.

Reads modulo's ``schedule_graph.json`` (which embeds the DDG), optionally augments
it with the pre-schedule ``ddg.json`` (occupancy + MinII bounds), formulates the
joint modulo-schedule + warp-assignment + memory problem as an OR-Tools CP-SAT
model (Twill: Soi et al., "Optimal Software Pipelining and Warp Specialization for
Tensor Core GPUs"), and rewrites an optimized ``schedule_graph.json`` that the
existing ``python -m sched2tlx`` emitter lowers to TLX.

The committed modulo schedule is always a feasible point, fed to the solver as a
warm-start hint and used as a no-regression floor: the solver never emits a
schedule worse than the baseline.
"""
