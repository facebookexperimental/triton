"""Empirical equivalence fuzzer + partition / soundness helpers.

This module is the *ground truth* against which a static equivalence checker is
measured. It is deliberately **standalone and project-agnostic** — it imports
only the Python standard library and never touches Triton, the PTX checker, or
GPUs. The only thing it needs from the caller is a ``run`` callable that maps a
config + random seed to the raw output bytes of that config:

    run(config, seed) -> bytes

Anything hashable can be a "config" (a namedtuple, a string, an int). The
fuzzer treats two configs as empirically equivalent iff they produce
*bit-identical* output on every random seed it tries. That partition of the
configs into equivalence classes is the ruler. A static checker's own partition
is then compared against it:

    * SOUND  — the checker partition *refines* the empirical one: every checker
               class sits inside a single empirical class. Equivalently: the
               checker never calls two configs equal that the fuzzer separated.
               The count of violating pairs is the **over-merge** count, and it
               must be 0.
    * RECOVERY GAP — how much *coarser* the empirical partition is (empirical
               classes the checker still splits into pieces). That is tuning
               freedom the checker leaves on the table — safe over-splitting,
               not a soundness bug.

A fuzzer can only ever *refute* equivalence (find a seed where the bits differ);
it can never prove it. So more seeds = stronger evidence, never certainty. The
two effort levels trade confidence for time:

    * "fast"       — 10 seeds.  A quick smoke / CI gate.
    * "convincing" — 1000 seeds. A strong empirical claim.

Reuse note: ported and generalized from the throwaway analysis scripts
(``overmerge_dump.py``, ``checker_vs_empirical.py``, ``setsize_eval.py``).
"""

import hashlib
import itertools
from collections import Counter

# Seed counts for the two fuzzer-effort levels (see module docstring).
EFFORT_REPEATS = {"fast": 10, "convincing": 1000}


def effort_repeats(effort):
    """Number of random seeds for a named fuzzer effort ("fast" / "convincing")."""
    try:
        return EFFORT_REPEATS[effort]
    except KeyError:
        raise ValueError(f"unknown fuzzer effort {effort!r}; expected one of {sorted(EFFORT_REPEATS)}")


def empirical_keys(run, configs, repeats, seed_start=0, progress=None):
    """Map each config to its empirical equivalence key.

    The key is the tuple of per-seed output digests over ``repeats`` random
    seeds. Two configs share a key iff their outputs are bit-identical on every
    seed, so grouping configs by this key yields the empirical partition. We
    store a SHA-1 digest per seed rather than the raw bytes purely to bound
    memory at high seed counts (digest equality matches byte equality up to a
    negligible collision probability).

    ``run(config, seed) -> bytes`` is the caller's GPU launch. ``progress``, if
    given, is called as ``progress(done, total)`` after each config.
    """
    keys = {}
    total = len(configs)
    for i, config in enumerate(configs, 1):
        digests = []
        for seed in range(seed_start, seed_start + repeats):
            digests.append(hashlib.sha1(run(config, seed)).digest())
        keys[config] = tuple(digests)
        if progress is not None:
            progress(i, total)
    return keys


def partition(configs, key_of):
    """Group ``configs`` into classes by ``key_of`` (a dict config -> hashable key).

    Returns a list of classes (each a list of configs), insertion-ordered so the
    output is deterministic.
    """
    groups = {}
    for config in configs:
        groups.setdefault(key_of[config], []).append(config)
    return list(groups.values())


def over_merge_pairs(configs, checker_key, empirical_key):
    """The exact config pairs the checker MERGES but the fuzzer SEPARATES.

    Each such pair is a soundness violation: the checker called them equal, yet
    some random input produced different bits. Returned for debugging; the count
    is what the soundness gate asserts is 0.
    """
    pairs = []
    for group in partition(configs, checker_key):
        for a, b in itertools.combinations(group, 2):
            if empirical_key[a] != empirical_key[b]:
                pairs.append((a, b))
    return pairs


def over_merges(configs, checker_key, empirical_key):
    """Count of over-merging pairs (soundness violations; must be 0).

    Computed per checker class without enumerating every pair: within a class of
    ``n`` configs, ``C(n,2)`` pairs are called equal; the empirically-justified
    ones are the same-key pairs ``sum C(m_i,2)``; the difference is the
    over-merge count.
    """
    total = 0
    for group in partition(configs, checker_key):
        n = len(group)
        same = sum(m * (m - 1) // 2 for m in Counter(empirical_key[c] for c in group).values())
        total += n * (n - 1) // 2 - same
    return total


def refines(configs, checker_key, empirical_key):
    """Does the checker partition refine the empirical one? (the formal soundness relation)

    Returns ``(holds, n_straddle)`` where ``n_straddle`` is the number of checker
    classes that straddle more than one empirical class. ``holds`` is True iff
    that is 0 — i.e. the checker never merges across an empirical boundary.
    """
    straddle = 0
    for group in partition(configs, checker_key):
        if len({empirical_key[c] for c in group}) > 1:
            straddle += 1
    return straddle == 0, straddle


def max_class_size(classes):
    """Size of the largest class in a partition (0 for an empty partition)."""
    return max((len(c) for c in classes), default=0)
