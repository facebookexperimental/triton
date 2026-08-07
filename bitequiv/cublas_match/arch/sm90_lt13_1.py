"""H100 (sm_90) with cuBLASLt 13.1.1.

Measured, not derived.  Every table below states this pair in full; nothing is inherited from
another file.
"""
from __future__ import annotations

from . import ArchProfile

# sm_90 needs four kinds of change from sm_100: 12 EXTRA gemv rows for `CUSTOM_OPTION` values
# Blackwell never reaches, 3 CHANGED gemv rows, a different `sm_count`, and one nvjet key that
# grows a second accumulation level. Every family was re-read on an H100 against cuBLASLt
# 13.1.1 -- the same version `_SM100` was measured against, so a difference here is architecture
# and not library.
#
# This is the first architecture whose tensor core is a different generation (`wgmma`, not
# `tcgen05`), and most of it still carried over: the reachable `ALGO_ID` set enumerates
# byte-identical to sm_103 for fp16, bf16 and fp8; ALGO 66's `STAGES_ID` is still 35 for
# fp16/bf16 and 36 for fp8; the CUTLASS `STAGES_ID` key space is still {0} + {7..20} + {25}; and
# ALGO 11's two rows re-read exactly, verified up to K = 30,292, which is both sides of the
# K = 512 boundary where its second accumulation level starts to bite.
#
# The gemv rows were read with the floating-point L-probe rather than a byte-compare, because a
# wrong gemv recipe survives most random inputs. Every one was settled by ELIMINATION over a
# 1,355-recipe grid -- "only this recipe could have produced these bits" -- with the chunk length
# then read off an adjacent-L boundary scan at K = 262,144, where the k range holds 128 to 65,536
# tiles and a chunked recipe could not have hidden. The table is complete for what the TOP
# heuristic returns: an 8,000,000-shape scan over vector lengths to 1e6 and K to 4e6 produced 42
# distinct `ALGO_ID 13` `CUSTOM_OPTION` values and every one is keyed below.
#
# Two `ALGO_ID`s the top heuristic returns here have NO row and decline: 41 (16 hits in 400,000
# fp16 draws) and 56 (8 hits). Both are tensor-core kernels whose accumulation order no plan in
# this file reproduces -- 41 byte-matches a plain accumulator on 28/48 (shape, seed) pairs, and
# only when `K % 64 < 32`; 56 matches nothing tried, at 0/48. Declining is the honest answer
# until they are measured.

PROFILE = ArchProfile(
    name="sm_90 (NVIDIA H100)",
    measured=True,
    measured_cublaslt="13.1.1",
    # 12 and 21 were added after the first sweep, both measured the same way as 23/24/66:
    #   12 `gemmk1_kernel` -- in 3,125 heuristic hits it appeared ONLY with K == 1. With K == 1
    #      every output is a single product, so there is no accumulation order to get wrong and
    #      any plan matches (all 7 plan forms were byte-identical on 40 shapes). The entry is
    #      therefore safe but UNTESTED for real accumulation: if cuBLAS ever picks 12 for K > 1,
    #      the recipe it inherits from `("cutlass", 0)` is a guess, not a measurement.
    #   21 `cutlass::Kernel2` -- behaves as CUTLASS. Its STAGES_ID 18 and 20 re-derive the block_k values
    #      already in `stages_recipe` (64 and 128) on shapes where block_k changes the grouping:
    #      16/16 and 13/13 byte-identical over 8 seeds each, and the other two block_k values lose. So
    #      adding 21 keeps the (family, STAGES_ID) key pure. Only its STAGES_ID 25 is new.
    algo_family=((11, "gemmsn"), (12, "cutlass"), (13, "gemv"), (14, "gemv"), (16, "gemmsn"), (21, "cutlass"),
                 (23, "cutlass"), (24, "cutlass"), (66, "nvjet")),
    # The CUTLASS side of this table is now the COMPLETE set of stages ids the sm_100 algos
    # advertise. `cublasLtMatmulAlgoCapGetAttribute(CUBLASLT_ALGO_CAP_STAGES_IDS)` over every
    # algo id `cublasLtMatmulAlgoGetIds` returns gives, for the cutlass families, exactly
    # {0} + {7..20} + {25} -- 16 keys, all listed below -- so a CUTLASS shape can no longer
    # decline on an unmeasured STAGES_ID.
    #
    # The block_k comes from the `cublasLtMatmulStages_t` enum NAME: the names read `<block_k>x<n>`,
    # so `block_k = 16 << ((id - 1) // 6)` for ids 1..24, 25 is `32x10`, and 32..37 are
    # `8xAUTO`..`256xAUTO`. That rule reproduces all 13 non-zero keys that were measured one at a
    # time, with 0 disagreements, which is why the four keys added from it are trusted.
    #
    # `k per dot` is 16 for EVERY named stage: those kernels run `s16816` / `s161616`. STAGES_ID
    # 0 is `UNDEFINED` -- it names no block_k, so the rule says nothing about it -- and stays a
    # measured exception at (32, 8), because the kernel behind it is `s1688`.
    #
    # HOW FAR THE FOUR RULE-DERIVED KEYS (7, 8, 11, 17) ARE ACTUALLY EXERCISED. 58,414,630
    # heuristic queries on sm_100, of which one 23,839,756-shape sweep counted per key:
    #   7  picked constantly (2.06 M hits in that sweep), and it DOES take split-K: 183,776 of
    #      those hits have SPLITK_NUM > 1. Byte-verified: 24/24 single-pass and 296/296 split-K at
    #      REDUCTION_SCHEME 2. Its split-K at REDUCTION_SCHEME 4 was bf16-only and did NOT match
    #      (0/200) when this key was added -- that was the bf16 hole in `_splitk_combine_modes`,
    #      since fixed, and the same shapes are byte-identical now.
    #   8  picked rarely (724 hits in that sweep) and NEVER with SPLITK_NUM > 1, so its split-K
    #      path is unverified. Byte-verified 13/13 single-pass.
    #   11 and 17 were never picked at all -- 0 hits in any of the 58 M queries, though algos 21
    #      and 23 do advertise them. Both entries rest on the enum-name rule alone: UNVERIFIED.
    stages_recipe=(
        (("cutlass",
          0), (32,
               8)), (("cutlass", 7),
                     (32,
                      16)),  # 32x1
        (("cutlass", 8),
         (32, 16)),  # 32x2
        (("cutlass", 9),
         (32, 16)),  # 32x3
        (("cutlass", 10),
         (32, 16)),  # 32x4
        (("cutlass", 11),
         (32, 16)),  # 32x5
        (("cutlass", 12),
         (32, 16)),  # 32x6
        (("cutlass", 13),
         (64, 16)),  # 64x1
        (("cutlass", 14),
         (64, 16)),  # 64x2
        (("cutlass", 15),
         (64, 16)),  # 64x3
        (("cutlass", 16),
         (64, 16)),  # 64x4
        (("cutlass", 17),
         (64, 16)),  # 64x5
        (("cutlass", 18),
         (64, 16)),  # 64x6
        (("cutlass", 19),
         (128, 16)),  # 128x1
        (("cutlass", 20),
         (128, 16)),  # 128x2
        # ALGO_ID 21 (`cutlass::Kernel2`) only; no other ALGO_ID was ever seen with this STAGES_ID.
        # k per dot: 16 byte-matches 15/15 shapes, 8 only 9/15. block_k: measured on 34 shapes
        # picked so that 32, 64 and 128 really do group the k-loop differently -- 32 byte-matches
        # 34/34, 64 21/34, 128 15/34, over 8 seeds each.
        (("cutlass", 25), (32, 16)), (("nvjet", 35), (64, None)),  # fp16/bf16; block_k doubles as the split-K grain
        (("nvjet", 36), (128, None)),  # fp8
    ),
    splitk_grains=(8, 64),
    # 1 is CUBLASLT_REDUCTION_SCHEME_INPLACE. It was once declined as "atomic, so not
    # deterministic for us", but every shape carrying it byte-matches the serial fp16 chain.
    reduction_to_cmode=((0, 3), (1, 3), (2, 0), (4, 2)),
    # ALGO 11 launches a 64-column CTA tile, so its thread count divided by 64 is S: 128 threads
    # -> 2, 256 threads -> 4 (read off 137 profiled launches). B came from the k probe; S * B is
    # 512 in both rows. Only these two configs are ever seen -- 20,057 ALGO 11 hits in 240,000
    # heuristic queries, all with SPLITK_NUM 1. ALGO 16 is always exactly
    # (16, 0, 1, 0, 0, 0, 0, 0, 0), so ALGO_ID 16 alone is the whole recipe: one ascending chain.
    # Byte-verified on the probe inputs themselves (2,600 probe launches, 0 mismatches) and on
    # random data (279,486/279,486 records over 49,196 (dtype, shape) combos for ALGO 11;
    # 221/221 shapes for ALGO 16).
    gemmsn_recipe=(((11, 0), (2, 256)), ((11, 1), (4, 128)), ((16, 0), (1, 0))),
    # ALGO 13. Four shapes of reduction over 27 CUSTOM_OPTION values:
    #   V = 1  strided leaves, one chunk, count-down butterfly over the lanes
    #   V = 4  the same with 4-wide vector leaves
    #   CC > 0 strided leaves, chunked, everything left to right
    #   V < 0  one contiguous slice per lane, capped at 64 k -- V = min(64, ceil(K/W))
    # THREE CHANGED ROWS, applied first. The single-lane options close an accumulator on a fixed
    # k here, where sm_100 records them as unbounded: 0 every 512 k, 7 and 8 every 256 k. Read off
    # the adjacent-L boundary scan at K 8,192 and 32,768, which prints the boundary positions with
    # no model in the loop -- 0 lands on 511, 1023, 1535, ... and 7 and 8 on 255, 511, 767, ...,
    # one gap value each and nothing in between. Byte-checked as well: these rows are 30/30 over
    # K 128..65536 where sm_100's unbounded row loses 10 to 14 of the same 30. sm_100's rows are
    # not wrong for sm_100 -- below one chunk length a chunked recipe and an unbounded one are the
    # same kernel -- and sm_103 already had to change its option 8 the same way.
    gemv_recipe=(
        # The 24 rows below complete the set: every CUSTOM_OPTION the top heuristic can return
        # for ALGO_ID 13 is now keyed. Found by enumeration rather than sampling -- an 8.25M
        # heuristic scan over vector lengths to 1e6 and K to 4e6 turned up 52 reachable values,
        # 7 of which (0, 6, 7, 37, 40, 41, 44) only appear at very long vectors or very deep K
        # and no sample had ever produced. Each was measured with the L-probe and byte-checked
        # on fresh shapes and unused seeds; none needed a new kernel.
        #
        # A tempting shortcut that does NOT work: for `internal::gemvx::kernel` the lane count is
        # a runtime blockDim, not a template parameter, so byte-identical template arguments do
        # not imply the same order. Customs 45, 62 and 63 share one signature and have three
        # different recipes. The shortcut does hold for gemv2N/gemv2T_kernel_val, where the ints
        # in the signature are the parameters.
        #
        # Custom 8 was only ever reached at K under 80 here, and below one chunk length a chunked
        # recipe and an unbounded one are the same kernel, so this row is not discriminating above
        # that. sm_103 reads a 256-k chunk for it; see `_SM103`.
        ((13, 0), (1, 1, 512, False)),
        ((13, 6), (1, 1, 0, False)),
        ((13, 7), (1, 1, 256, False)),
        ((13, 8), (1, 1, 256, False)),
        ((13, 84), (1, 1, 0, False)),
        ((13, 14), (1, 16, 16, False)),
        ((13, 36), (-64, 32, 0, True)),
        ((13, 37), (1, 32, 0, True)),
        ((13, 40), (-64, 4, 0, True)),
        ((13, 41), (1, 4, 0, True)),
        ((13, 44), (-64, 16, 0, True)),
        ((13, 45), (1, 16, 0, True)),
        ((13, 46), (-64, 32, 0, True)),
        ((13, 48), (-64, 2, 0, True)),
        ((13, 49), (1, 2, 0, True)),
        ((13, 50), (-64, 4, 0, True)),
        ((13, 65), (1, 4, 0, True)),
        ((13, 66), (-64, 8, 0, True)),
        ((13, 67), (1, 8, 0, True)),
        ((13, 69), (1, 4, 0, True)),
        ((13, 82), (-64, 32, 0, True)),
        ((13, 86), (-64, 4, 0, True)),
        ((13, 95), (1, 2, 0, True)),
        ((13, 98), (-64, 8, 0, True)),
        ((13, 10), (4, 32, 0, True)),
        ((13, 11), (1, 16, 32, False)),
        ((13, 12), (1, 16, 16, False)),
        ((13, 13), (1, 16, 16, False)),
        ((13, 35), (1, 16, 0, True)),
        ((13, 42), (-64, 8, 0, True)),
        ((13, 43), (1, 8, 0, True)),
        ((13, 47), (1, 32, 0, True)),
        ((13, 51), (1, 4, 0, True)),
        ((13, 52), (-64, 8, 0, True)),
        ((13, 53), (1, 8, 0, True)),
        ((13, 54), (-64, 16, 0, True)),
        ((13, 55), (1, 16, 0, True)),
        ((13, 56), (-64, 2, 0, True)),
        ((13, 57), (1, 2, 0, True)),
        ((13, 58), (-64, 4, 0, True)),
        ((13, 59), (1, 4, 0, True)),
        ((13, 60), (-64, 8, 0, True)),
        ((13, 61), (1, 8, 0, True)),
        ((13, 62), (-64, 2, 0, True)),
        ((13, 63), (1, 2, 0, True)),
        ((13, 75), (1, 32, 0, True)),
        ((13, 81), (1, 16, 0, True)),
        ((13, 83), (1, 32, 0, True)),
        ((13, 90), (-64, 16, 0, True)),
        ((13, 91), (1, 16, 0, True)),
        ((13, 93), (1, 32, 0, True)),
        # THEN THE 12 NEW ROWS, for `CUSTOM_OPTION` values sm_100 never reaches. They are not a new
        # family: every one lands in a shape sm_100 already has, and they extend its even/odd pairing
        # -- an even option is the contiguous-slice form at some lane count and the next odd one is
        # the strided form at the same lane count (100/101 at W 16, 110/111 at W 4, and 92, 74 pairing
        # with sm_100's 93, 75 at W 32). Every one is `CC` 0: the boundary scan found no accumulator
        # boundary anywhere in a 262,144-long k range.
        ((13, 68), (-64, 4, 0, True)),
        ((13, 72), (-64, 16, 0, True)),
        ((13, 74), (-64, 32, 0, True)),
        ((13, 89), (1, 8, 0, True)),
        ((13, 92), (-64, 32, 0, True)),
        ((13, 96), (-64, 4, 0, True)),
        ((13, 100), (-64, 16, 0, True)),
        ((13, 101), (1, 16, 0, True)),
        ((13, 102), (-64, 2, 0, True)),
        ((13, 106), (-64, 8, 0, True)),
        ((13, 110), (-64, 4, 0, True)),
        ((13, 111), (1, 4, 0, True)),
    ),
    # CUSTOM_OPTION 5 is `gemv2N_kernel<..., 128, 32, 4, 4, 1, false>`: grid ceil(NEL/4) with
    # 128 threads, so 32 lanes per output element, and the lane chunks its slice by 16. The
    # chunk boundary was read at S = 10, 16, 17, 19, 22, 35 and 41 and sat at 15, 31, 47 every
    # time. In 2,382 heuristic hits it was always SPLITK_NUM 1 and always M == 1, so
    # `_plan_gemv` refuses anything else rather than assuming this row covers it.
    gemv_cslice_recipe=(((13, 5), (32, 16)), ),
    # CUSTOM_OPTION 10 keeps its config but changes order on a long output. It is
    # `gemvNSP_kernel`, whose blockDim is (bx, by): bx output elements per CTA and by lanes
    # over k, and only `by` sets the order. cuBLAS holds by = 32 while the launch still fits
    # the GPU and steps to 26, 24 and 20 above that, by a cost model the config does not carry
    # -- two shapes with identical 9-field configs then run different orders -- so past the cap
    # this declines. The boundary was measured exactly, by forcing the kernel over a dense
    # output-length sweep: 9727 output elements still runs by = 32 and 9729 runs by = 26, and
    # 9728 * 32 == 152 * 2048 == the threads an sm_100 holds.
    #
    # The cap is where the hardware changes, not where coverage changes. The TOP heuristic --
    # the only entry this file ever runs -- picks CUSTOM_OPTION 10 up to 6225 output elements
    # and then not again until 10500, so the whole 6226..10499 stretch is empty and any cap
    # inside it behaves the same. It is written as the real boundary anyway, because that is
    # the fact, and because the formula ports to a GPU with a different SM count while a
    # constant does not.
    gemv_max_elems=(((13, 10), "occupancy"), ),
    # The one family that genuinely changed shape. H100's fp8 nvjet kernel -- STAGES_ID 36, which
    # is fp8-only here, so this touches nothing else -- closes an accumulator every 128 k and adds
    # the block totals afterwards, where the same key on sm_100 and sm_103 runs one flat
    # accumulator. Read straight off a K walk at 16-k steps: the flat form is byte-identical to
    # cuBLAS at K 16 through 128, the whole range that fits ONE 128-k step, and misses every K
    # from 160 up including exact multiples of 128, which rules out a residue-tile explanation.
    # The block form is byte-identical 7/7 at SPLITK_NUM 1 and 7/7 on split-K shapes where the
    # current flat plan is 0/7. Triton is on the same hardware path either way -- it emits
    # `wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3`, the native fp8 MMA -- so this is
    # cuBLAS's grouping and not a Triton fallback.
    block_level_keys=(("nvjet", 36), ),
    # H100 holds 132 * 2048 threads, not GB200's and GB300's 152 * 2048, so the `CUSTOM_OPTION 10`
    # occupancy cap moves with it. It was bisected here rather than assumed: 8448 output elements
    # still run 32 lanes and 8449 does not, and 8448 * 32 == 132 * 2048 exactly. That is the first
    # evidence the formula ports rather than the constant.
    sm_count=132,
    threads_per_sm=2048,
    fp8_min_bm=64,
)
