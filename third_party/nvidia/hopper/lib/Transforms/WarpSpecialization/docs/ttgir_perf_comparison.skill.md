# Triaging TTGIR Performance Differences

This skill provides a systematic workflow for comparing two TTGIR files that
implement the same kernel functionality (e.g., autoWS vs TLX) and identifying
the root causes of performance gaps.

## When to Use

- You have two TTGIR files for the same kernel (e.g., autoWS and TLX flash attention)
- One version is faster than the other
- You need to identify which IR differences cause the performance gap

## Prerequisites

- Two TTGIR files to compare (the "slow" version and the "fast" reference)
- Optional: benchmark output showing the performance gap

## Workflow

### Step 1: Quick Structural Comparison

Run these commands on both TTGIR files to get a high-level overview of differences:

```bash
A=<slow.ttgir>
B=<fast.ttgir>

echo "=== File sizes ==="
wc -l $A $B

echo "=== convert_layout count ==="
echo "  slow: $(grep -c convert_layout $A)"
echo "  fast: $(grep -c convert_layout $B)"

echo "=== Barrier count ==="
echo "  slow: init=$(grep -c init_barrier $A) wait=$(grep -c wait_barrier $A) arrive=$(grep -c arrive_barrier $A) gpu=$(grep -c gpu.barrier $A)"
echo "  fast: init=$(grep -c init_barrier $B) wait=$(grep -c wait_barrier $B) arrive=$(grep -c arrive_barrier $B) gpu=$(grep -c gpu.barrier $B)"

echo "=== join/split/subslice ==="
echo "  slow: join=$(grep -c 'tt.join' $A) split=$(grep -c 'tt.split' $A) subslice=$(grep -c tmem_subslice $A)"
echo "  fast: join=$(grep -c 'tt.join' $B) split=$(grep -c 'tt.split' $B) subslice=$(grep -c tmem_subslice $B)"

echo "=== scf.if (conditional branches) ==="
echo "  slow: $(grep -c 'scf.if' $A)"
echo "  fast: $(grep -c 'scf.if' $B)"

echo "=== Register allocation ==="
grep "requestedRegisters\|max_reg\|min_reg" $A
grep "requestedRegisters\|max_reg\|min_reg" $B

echo "=== TMEM encodings ==="
grep "tensor_memory_encoding" $A
grep "tensor_memory_encoding" $B

echo "=== 3D blocked layouts (red flag for join/trans/reshape) ==="
echo "  slow: $(grep -c 'sizePerThread = \[1, .*, .*\]' $A)"
echo "  fast: $(grep -c 'sizePerThread = \[1, .*, .*\]' $B)"
```

Use the output to guide which areas to investigate in depth.

### Step 2: Check Each Difference Category

Investigate each category below in order of typical impact. Read the relevant
sections of both TTGIR files and compare them.

---

## Difference Categories (Ordered by Typical Impact)

### Category 1: Data Extraction — split/join vs tmem_subslice

**Impact: 🔴 CRITICAL (10-15% TFLOPS)**

This is the single most impactful pattern found in practice. When accumulator
data needs to be split into halves (for epilogue TMA stores or accumulator
rescaling), the compiler can generate two very different patterns:

**Slow pattern (load direction):**
```
%load = tmem_load %src : memdesc<MxN>
%r = tt.reshape %load : tensor<MxN> -> tensor<Mx2x(N/2)>
%t = tt.trans %r {order=[0,2,1]}
%c = ttg.convert_layout %t          ← expensive cross-thread shuffle
%a, %b = tt.split %c
```

**Fast pattern (load direction):**
```
%s0 = tmem_subslice %src {N=0}      ← zero-cost TMEM view
%s1 = tmem_subslice %src {N=N/2}
%a = tmem_load %s0 : memdesc<Mx(N/2)>
%b = tmem_load %s1 : memdesc<Mx(N/2)>
```

**Slow pattern (store direction):**
```
%j = tt.join %a, %b                 ← 3D intermediate
%t = tt.trans %j {order=[0,2,1]}    ← 3D intermediate
%r = tt.reshape %t : -> tensor<MxN>
tmem_store %r, %dst : memdesc<MxN>  ← full-width store
```

**Fast pattern (store direction):**
```
%s0 = tmem_subslice %dst {N=0}
%s1 = tmem_subslice %dst {N=N/2}
tmem_store %a, %s0                  ← half-width in-place store
tmem_store %b, %s1                  ← half-width in-place store
```

**What to grep for:**
```bash
grep -n "tt.join\|tt.split\|tmem_subslice\|tt.reshape.*blocked\|tt.trans.*order" file.ttgir
```

**Red flags:**
- `tt.join` or `tt.split` in the slow file that don't appear in the fast file
- 3D `#blocked` layouts (e.g., `sizePerThread = [1, 64, 2]`) — these only exist
  to support join/trans/reshape chains
- `convert_layout` between 3D layouts

**Validated fix:** The `TMemSplitLoadPattern` in `OptimizeTMemLayouts.cpp` handles
the load-direction pattern for M=64 and M=128. The store-direction pattern (join)
requires a separate optimization or TTGIR override.

---

### Category 2: Conditional vs Unconditional Masking (Causal Attention)

**Impact: 🔴 CRITICAL for causal kernels (masking ~97% of iterations is wasted)**

For causal attention, only K-tiles on or above the diagonal need masking.
Tiles fully below the diagonal have no elements to mask.

**Slow pattern (unconditional masking):**
```
scf.for %k = 0 to %num_k_tiles {
  // masking applied on EVERY iteration
  %masked = map_elementwise(%qk, ...) {
    %mask = arith.shli %c-1, %col_lim
    %bit  = arith.shli %c1, %bit_pos
    %test = arith.andi %mask, %bit
    %cmp  = arith.cmpi eq, %test, %c0
    %out  = arith.select %cmp, %val, %neg_inf
  }
}
```

**Fast pattern (conditional masking):**
```
scf.for %k = 0 to %num_k_tiles {
  %needs_mask = arith.cmpi sge, %k_offset, %m_offset
  %masked = scf.if %needs_mask {
    // masking only for diagonal tiles
    %m = map_elementwise(%qk, ...) { ... bitmask logic ... }
    scf.yield %m
  } else {
    scf.yield %qk   // pass through unchanged
  }
}
```

For seq_len=8192 with block_size=128, each M-tile has ~64 K-tiles but only 1-2
are on the diagonal. The unconditional pattern wastes the expensive
`map_elementwise` bitmask computation on ~62 fully-valid tiles.

**What to grep for:**
```bash
# Check for conditional masking
grep -n "scf.if\|cmpi sge\|cmpi slt" file.ttgir

# Check for masking ops
grep -n "map_elementwise\|select.*0xFF800000\|shli.*c-1" file.ttgir
```

**Red flag:** The slow file has `map_elementwise` with bitmask ops but no
`scf.if` guard, while the fast file wraps masking in `scf.if`.

---

### Category 3: Register Allocation

**Impact: 🟡 MEDIUM-HIGH (can cause spills → local memory access)**

```bash
grep "requestedRegisters\|max_reg_auto_ws\|min_reg_auto_ws" file.ttgir
```

The register budget is split across WS partitions. Key things to compare:
- **Computation partition registers**: This partition runs softmax, reductions,
  masking, and FMA chains. It needs the most registers.
- **Gemm/Load partition registers**: Typically need fewer (24-48).
- Total budget is constrained to 256 regs per thread.

Example:
```
# autoWS: computation gets 152 regs
ttg.max_reg_auto_ws = 152, ttg.min_reg_auto_ws = 24

# TLX: computation gets 168 regs
requestedRegisters = array<i32: 168, 168, 24, 24, 24>
```

16 fewer registers can cause spills to local memory (high-latency L2 access).

---

### Category 4: Accumulator Zero Initialization

**Impact: 🟡 MEDIUM**

```bash
grep -n "tmem_store.*dense<0\|tmem_store.*cst.*0.0\|useAccumulator" file.ttgir
```

| Pattern | Cost |
|---------|------|
| `tmem_store dense<0.0>` per outer tile | Extra TMEM store bandwidth per iteration |
| `useAccumulator=false` on first MMA | Free — MMA hardware zeros the accumulator |

In autoWS, the compiler may generate explicit zero stores because it doesn't
know the first MMA will clear the accumulator. TLX can explicitly set
`use_acc=False` on the first dot.

---

### Category 5: convert_layout Count and Cost

**Impact: 🟡 MEDIUM (each may use shared memory for shuffling)**

```bash
grep -c "convert_layout" file.ttgir
```

Each `convert_layout` between incompatible `#blocked` layouts typically goes
through shared memory (alloc → store → barrier → load). Compare:
- Total count in each file
- Whether they appear inside the inner loop (much more costly)
- Whether they convert between 2D and 3D layouts (usually from join/split chains)

---

### Category 6: TMEM Encodings

**Impact: 🟢 LOW-MEDIUM**

```bash
grep "tensor_memory_encoding" file.ttgir
```

Compare `blockM`, `blockN`, and `colStride` values:
- `colStride=1` is standard (contiguous column access)
- `colStride=2` means strided access (used for f32→bf16 reinterpret) and may
  cause TMEM bank conflicts during MMA reads
- Different `blockM`/`blockN` affect layout compatibility with MMA ops

---

### Category 7: Barrier and Synchronization Overhead

**Impact: 🟢 LOW-MEDIUM (amortized in persistent kernels)**

```bash
grep -c "init_barrier\|wait_barrier\|arrive_barrier\|gpu.barrier" file.ttgir
```

Compare:
- **`gpu.barrier` count**: Full CTA sync, expensive — serializes all warps.
  Should be 0 in the steady-state loop.
- **Individual vs array barriers**: `memdesc<1xi64>` vs `memdesc<3xi64>`.
  Array barriers use less SMEM.
- **Barriers inside loops vs outside**: Loop-interior barriers cost more.

---

### Category 8: Loop Structure and Conditional Branches

**Impact: Varies**

```bash
grep -n "scf.for\|scf.if\|scf.yield" file.ttgir
```

Compare:
- Loop bounds and step sizes
- Loop nesting depth
- `scf.if` guards (one file may skip work the other does unconditionally)
- Loop attributes (`tt.disallow_acc_multi_buffer`, `tt.warp_specialize`)
- Persistent kernel tile scheduling (swizzled vs linear)

---

### Category 9: Warp Specialization Partition Structure

**Impact: Varies**

```bash
grep "async_task_id\|warp_specialize\|requestedRegisters" file.ttgir | head -20
```

Compare:
- Number of partitions and their roles (computation, gemm, load, store, epilogue)
- Whether there's a **dedicated store partition** (TLX often has one)
- How many warps each partition gets
- Which ops end up in which partition

---

### Category 10: Memory Access Patterns

**Impact: 🟢 LOW**

```bash
grep -n "descriptor_load\|descriptor_store\|descriptor_reduce\|local_load\|local_store\|async_tma" file.ttgir
```

Compare:
- TMA descriptor ops: same count and types?
- Whether stores use `descriptor_reduce` (TMA atomic reduce-add) vs regular stores
- Async pipeline depth (number of buffered stages)
- Total shared memory allocation sizes

---

## Step 3: Create TTGIR Override to Validate Fixes

Once you identify a difference, create a TTGIR override to validate its impact.
See the `debugging_accuracy.skill.md` skill (Option 4) for the full override
workflow using `TRITON_KERNEL_DUMP` and `TRITON_KERNEL_OVERRIDE`.

Quick summary:
1. Dump: `TRITON_KERNEL_DUMP=1 python kernel.py` → `~/.triton/dump/<hash>/`
2. Copy and modify: `cp dump/<hash>/kernel.ttgir override/<hash>/kernel.ttgir`
3. Apply the fix manually to the override TTGIR
4. Test: `TRITON_KERNEL_OVERRIDE=1 python kernel.py`
5. Compare performance with and without the override

## Case Studies

### BWD Flash Attention: dq reshape+split → tmem_subslice (Category 1)
- **Gap**: autoWS 756 TFLOPS vs TLX 826 TFLOPS (0.89x)
- **Root cause**: `tmem_load(64x128) → reshape → trans → convert_layout → split`
- **Fix**: Extended `TMemSplitLoadPattern` in `OptimizeTMemLayouts.cpp` to M=64
- **Result**: 831 TFLOPS (1.01x TLX), +10-12% improvement

### FWD Causal Flash Attention: unconditional masking + join+trans+reshape (Categories 1+2)
- **Root cause 1**: Causal bitmask applied on every K-tile (Category 2)
- **Root cause 2**: Acc rescaling uses `join → trans → reshape → tmem_store` (Category 1)
- **Status**: Analysis complete, override TTGIR prepared
