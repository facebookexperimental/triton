# Memory Planner Visualization

This document describes the visualization tools for debugging the Warp Specialization memory planner. The visualizations help understand buffer liveness, channel dependencies, and data flow between partitions.

## What's Implemented

### 1. SMEM Buffer Liveness (`dumpSmemBufferLiveness`)
Visualizes shared memory buffer allocations with:
- Buffer names extracted from source locations
- Liveness intervals `[start-end)` based on operation IDs
- Buffer sizes in bytes
- Channel associations

### 2. TMEM Buffer Liveness (`dumpTmemBufferLiveness`)
Visualizes tensor memory buffer allocations with:
- Buffer names extracted from source locations
- **Row × Column dimensions** (e.g., `128x128`, `128x64`, `128x1`)
- Liveness intervals `[start-end)` based on operation IDs
- Channel count per buffer
- OperandD flag for accumulator buffers
- Summary table with all buffer information

### 3. Combined Key Ops + Channel Graph (`dumpCombinedGraph`)
Visualizes the complete dataflow structure:
- Operations grouped by partition (async task ID)
- Vertical program order within each partition
- Channel edges showing data dependencies:
  - **Green edges**: SMEM channels
  - **Red edges**: TMEM channels
- Operation shapes and types (loads, stores, MMA, etc.)

## How to Dump DOT Files

### Method 1: Using Environment Variable (Recommended)

Set `TRITON_DUMP_WS_GRAPHS` to a directory path to automatically dump DOT files:

```bash
# Create output directory
mkdir -p /tmp/ws_graphs

# Run with environment variable
TRITON_DUMP_WS_GRAPHS=/tmp/ws_graphs \
TRITON_USE_META_WS=1 \
python your_test.py

# Files will be created:
# /tmp/ws_graphs/smem_liveness_0.dot
# /tmp/ws_graphs/tmem_liveness_1.dot
# /tmp/ws_graphs/combined_graph_2.dot
```

```bash
# Clean and render to PNG (strip header/footer markers)
sed -n '/^digraph/,/^}$/p' /tmp/ws_graphs/smem_liveness_0.dot | dot -Tpng -o /tmp/ws_graphs/smem_liveness.png
sed -n '/^digraph/,/^}$/p' /tmp/ws_graphs/tmem_liveness_2.dot | dot -Tpng -o /tmp/ws_graphs/tmem_liveness.png
sed -n '/^digraph/,/^}$/p' /tmp/ws_graphs/combined_graph_1.dot | dot -Tpng -o /tmp/ws_graphs/combined.png

# Combine all three into one image
convert /tmp/ws_graphs/smem_liveness.png /tmp/ws_graphs/tmem_liveness.png \
        /tmp/ws_graphs/combined.png -append /tmp/ws_graphs/all.png
```

### Method 2: Extract from Debug Output

#### Step 1: Build with Debug Support
```bash
pip install -e . --no-build-isolation
```

#### Step 2: Run with Debug Flags
```bash
TRITON_LLVM_DEBUG_ONLY="nvgpu-ws-memory-planner" \
MLIR_ENABLE_DUMP=1 \
python your_test.py 2>&1 | tee output.txt
```

### Step 3: Extract DOT Files
```bash
# Extract SMEM liveness graph
awk '/=== SMEM Buffer Liveness Graph ===/,/=== End SMEM Buffer Liveness Graph ===/' \
  output.txt | sed -n '2,/=== End/p' | head -n -1 > smem_liveness.dot

# Extract TMEM liveness graph
awk '/=== TMEM Buffer Liveness Graph ===/,/=== End TMEM Buffer Liveness Graph ===/' \
  output.txt | sed -n '2,/=== End/p' | head -n -1 > tmem_liveness.dot

# Extract Combined graph
awk '/=== Combined Key Ops \+ Channel Graph/,/=== End Combined Graph ===/' \
  output.txt | grep -v "=== Combined" | grep -v "// Render with" | head -n -1 > combined.dot
```

### Step 4: Render to PNG
```bash
dot -Tpng smem_liveness.dot -o smem_liveness.png
dot -Tpng tmem_liveness.dot -o tmem_liveness.png
dot -Tpng combined.dot -o combined.png
```

## Combining All Plots into One Image

Use Python with PIL to combine the three images:

```python
from PIL import Image

# Load images
smem_img = Image.open('smem_liveness.png')
tmem_img = Image.open('tmem_liveness.png')
combined_img = Image.open('combined.png')

# Calculate dimensions
max_width = max(smem_img.width, tmem_img.width, combined_img.width)
total_height = smem_img.height + tmem_img.height + combined_img.height + 60  # 60px for labels

# Create combined image
result = Image.new('RGB', (max_width, total_height), 'white')

# Paste images vertically
y_offset = 0
result.paste(smem_img, (0, y_offset))
y_offset += smem_img.height + 20

result.paste(tmem_img, (0, y_offset))
y_offset += tmem_img.height + 20

result.paste(combined_img, (0, y_offset))

# Save
result.save('memory_planner_visualization.png')
print(f"Saved combined image: {max_width}x{total_height}")
```

Or use ImageMagick for a quick combination:
```bash
convert smem_liveness.png tmem_liveness.png combined.png -append memory_planner_all.png
```

## Output Example

### SMEM Buffer Liveness
Shows buffers like:
- `dq 49152 [0-42)` - 48KB buffer, live from op 0 to op 42
- `do 32768 [5-38)` - 32KB buffer, live from op 5 to op 38

### TMEM Buffer Liveness
Shows buffers with dimensions:
| Name | Size | Channels | Liveness | OperandD |
|------|------|----------|----------|----------|
| dk | 128x128 | 2 | [44-98) | 2 |
| dv | 128x128 | 2 | [45-96) | 2 |
| qkT | 128x128 | 1 | [56-61) | 0 |
| dpT | 128x128 | 1 | [73-78) | 0 |

### Combined Graph
Shows partitions with operations in program order:
- **Partition 0** (blue): Global loads
- **Partition 1** (green): SMEM stores, MMA producers
- **Partition 4/5** (red/yellow): Compute partitions
- **Partition 3**: Final stores

Channel edges show:
- Green arrows: SMEM data transfers
- Red arrows: TMEM data transfers (including OperandD accumulators)

## Epilogue Buffer Fusion

### What It Does

When a single `tmem_load` result is split into multiple sub-tiles that are stored to separate SMEM buffers (the epilogue pattern), these buffers are used sequentially with disjoint liveness. The epilogue buffer fusion optimization detects this pattern and assigns the same `buffer.id` to all such buffers so they share physical SMEM, reducing overall shared memory consumption.

### How It Works

The algorithm follows the same logical steps in both code paths:

1. **Group buffers by original load op.** For each candidate buffer, trace back through its channel's `LocalStoreOp` source using `findOriginalLoadOp`, which walks backward through transparent ops (`SplitOp`, `ReshapeOp`, `TransOp`, `ConvertLayoutOp`, truncation/extension casts, `BitcastOp`) to find the root `TMEMLoadOp`. Buffers that originate from the same `TMEMLoadOp` are grouped together.

2. **Skip small groups.** Groups with fewer than 2 buffers have nothing to fuse.

3. **Check compatibility.** All allocs in the group must have the same element type and SMEM size (checked by `allAllocsCompatible`).

4. **Verify disjoint liveness** (legacy path only). Buffers are sorted by liveness start, then all pairs are checked for overlap. If any intervals overlap, the group is skipped.

5. **Assign shared buffer ID.** All buffers in the group receive the same `buffer.id` (or `bufferId`), so they share the same physical SMEM allocation.

### Two Code Paths

| Aspect | Legacy (`fuseEpilogueBuffers`) | New (`fuseEpilogueWSBuffers`) |
|--------|-------------------------------|-------------------------------|
| Phase | Phase 2 of `MemoryPlanner::run()` | Phase 3.5 of `allocateSmemBuffers()` |
| Scope | `MemoryPlanner` member function | Free function in anonymous namespace |
| Buffer filter | Non-innermost-loop buffers | `P2_Other` priority WSBuffers |
| Liveness check | Pairwise disjoint verification (with sort) | None (sequential use assumed by priority classification) |

### Debugging

Enable debug logging with:

```bash
TRITON_LLVM_DEBUG_ONLY="nvgpu-ws-memory-planner" python your_test.py 2>&1
```

Look for these messages:
- `"Phase 2 (epilogue fusion): merged N buffers into buffer.id=X"` — legacy path
- `"Phase 3.5 (epilogue fusion): merged N P2_Other buffers into bufferId=X"` — new path

### Limitations

The optimization does not yet support increasing the buffer count in the epilogue (i.e., it only fuses existing buffers but cannot create additional copies for deeper pipelining of epilogue stores).

## Source Files

- **Declaration**: `CodePartitionUtility.h`
- **Implementation**: `CodePartitionUtility.cpp`
- **Call sites**: `WSMemoryPlanner.cpp` (in `MemoryPlanner::run()` and `MemoryPlannerTmem::run()`)

## Debug Flags Reference

| Flag | Purpose |
|------|---------|
| `TRITON_DUMP_WS_GRAPHS=/path/to/dir` | **Dump DOT files directly to directory** (recommended) |
| `TRITON_LLVM_DEBUG_ONLY="nvgpu-ws-memory-planner"` | Enable memory planner debug output to stderr |
| `MLIR_ENABLE_DUMP=1` | Enable MLIR pass dumps |
| `TRITON_USE_META_WS=1` | Use Meta's warp specialization passes |
