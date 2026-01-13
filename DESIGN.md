# Numerical Invariance Integration - Design Document

This document is the source of truth for integrating numerical invariance fingerprinting into the Triton compilation pipeline.

## Overview

**Goal**: Automatically compute numerical invariance fingerprints for TTGIR modules to detect regressions that could cause numerical mismatches.

**Architecture**: Two-phase hook system
- **Phase 1** (`make_ttgir`): Cache invariance for ALL configs being autotuned
- **Phase 2** (autotuner): Store only the WINNER's invariance to baselines

**Environment Variables**:
- `TRITON_INVARIANCE_COMMIT`: Write winner's invariance to baselines
- `TRITON_INVARIANCE_COMPARE`: Compare against baselines, raise on mismatch

**Storage**: `baselines/numerical_invariance/{kernel_name}_{sig_hash}.json`

---

## Phase Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE DEPENDENCY GRAPH                            │
│                                                                             │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│  │ Phase 1  │     │ Phase 2  │     │ Phase 3  │     │ Phase 4  │          │
│  │ Pybind   │     │ Data     │────▶│ Cache    │     │ Storage  │          │
│  │ Binding  │     │ Layer    │     │ Layer    │     │ Layer    │          │
│  └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘          │
│       │                │                │                │                 │
│       │                └───────┬────────┴────────────────┘                 │
│       │                        │                                           │
│       │                        ▼                                           │
│       │              ┌──────────────────┐                                  │
│       └─────────────▶│     Phase 5      │                                  │
│                      │  Integration API │                                  │
│                      └────────┬─────────┘                                  │
│                               │                                            │
│                ┌──────────────┼──────────────┐                             │
│                ▼              │              ▼                             │
│         ┌──────────┐          │       ┌──────────┐                         │
│         │ Phase 6  │          │       │ Phase 7  │                         │
│         │make_ttgir│──────────┼──────▶│autotuner │                         │
│         │  Hook    │          │       │  Hook    │                         │
│         └──────────┘          │       └────┬─────┘                         │
│                               │            │                               │
│                               ▼            ▼                               │
│                         ┌──────────────────────┐                           │
│                         │       Phase 8        │                           │
│                         │   E2E Integration    │                           │
│                         └──────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Recommended Implementation Order**: 2 → 3 → 4 → 1 → 5 → 6 → 7 → 8

(Phases 2-4 are pure Python, no compilation needed. Phase 1 requires rebuild. This ordering minimizes rebuild cycles during development.)

---

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            AUTOTUNER LOOP                                │
│  for config in configs:                                                  │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │  make_ttgir(src, opt=config)                                   │   │
│    │    └─→ pm.run(mod)                                             │   │
│    │    └─→ cache_invariance(mod, name, sig, cfg) ← PHASE 1 HOOK   │   │
│    │    └─→ return mod                                              │   │
│    └────────────────────────────────────────────────────────────────┘   │
│    benchmark(config) → timings[config] = time                           │
│                                                                          │
│  winner = min(timings, key=timings.get)                                 │
│  self.cache[key] = winner                                               │
│  on_winner_selected(name, sig, cfg_hash(winner)) ← PHASE 2 HOOK        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

| Decision | Resolution |
|----------|------------|
| Cache eviction | Clear after autotuner finishes each kernel |
| Signature hash | Types only (no tensor shapes) |
| Error handling | **Raise** on compute failures |
| Kernel name access | `mod.get_entry_func_name()` in make_ttgir |

---

## Phase 1: Pybind Binding

### Goal
Expose C++ `computeNumericalInvariance()` to Python via pybind11.

### Inputs
- Existing C++ implementation in `lib/Dialect/TritonGPU/Transforms/NumericalInvariance.cpp`
- Existing header `include/triton/Dialect/TritonGPU/Transforms/NumericalInvariance.h`

### Outputs
- `triton._C.libtriton.passes.ttgpuir.compute_numerical_invariance(mod) -> dict`

### Files Modified
- `python/src/passes.cc`

### Test Strategy
```python
# test_pybind_invariance.py
def test_compute_numerical_invariance_binding_exists():
    """Verify the binding is accessible."""
    from triton._C.libtriton import passes
    assert hasattr(passes.ttgpuir, 'compute_numerical_invariance')

def test_compute_numerical_invariance_returns_dict():
    """Verify return type and keys."""
    # Create minimal TTGIR module via triton.jit compilation
    @triton.jit
    def add_kernel(x_ptr, y_ptr, n: tl.constexpr):
        tl.store(y_ptr, tl.load(x_ptr))
    
    # Compile to TTGIR, call binding, verify dict structure
    result = passes.ttgpuir.compute_numerical_invariance(mod)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'fingerprint', 'computation_dag_hash', 
                                   'dtype_signature', 'layout_hash', 'hw_config_hash'}

def test_same_kernel_same_fingerprint():
    """Verify determinism."""
    # Compile same kernel twice, verify fingerprints match
```

### Rollback Plan
- Revert `passes.cc` changes (single file)
- No other files affected

---

## Phase 2: Data Layer

### Goal
Define `InvarianceData` dataclass with serialization/deserialization.

### Inputs
- None (independent)

### Outputs
- `InvarianceData` dataclass
- `to_dict()` / `from_dict()` methods
- `diff_report()` method

### Files Created
- `python/triton/runtime/invariance.py` (partial)

### Test Strategy
```python
# test_invariance_data.py
def test_invariance_data_creation():
    inv = InvarianceData(fingerprint=123, computation_dag_hash=456, 
                         dtype_signature="f16->f32", layout_hash=789, hw_config_hash=101)
    assert inv.fingerprint == 123

def test_invariance_data_to_dict_roundtrip():
    inv = InvarianceData(...)
    d = inv.to_dict()
    inv2 = InvarianceData.from_dict(d)
    assert inv == inv2

def test_invariance_data_equality():
    inv1 = InvarianceData(fingerprint=100, ...)
    inv2 = InvarianceData(fingerprint=100, ...)
    inv3 = InvarianceData(fingerprint=200, ...)
    assert inv1 == inv2
    assert inv1 != inv3

def test_diff_report_shows_differences():
    inv1 = InvarianceData(fingerprint=1, computation_dag_hash=100, ...)
    inv2 = InvarianceData(fingerprint=2, computation_dag_hash=200, ...)
    report = inv1.diff_report(inv2)
    assert "computation_dag_hash" in report
```

### Rollback Plan
- Delete `invariance.py` (single file)

---

## Phase 3: Cache Layer

### Goal
Thread-safe `InvarianceCache` for storing invariances during autotuning.

### Inputs
- Phase 2: `InvarianceData`

### Outputs
- `InvarianceCache` class with `put()`, `get()`, `clear_kernel()`
- Global `_invariance_cache` instance

### Files Modified
- `python/triton/runtime/invariance.py`

### Test Strategy
```python
# test_invariance_cache.py
def test_cache_put_and_get():
    cache = InvarianceCache()
    inv = InvarianceData(...)
    cache.put("kernel", "sig123", "cfg456", inv)
    result = cache.get("kernel", "sig123", "cfg456")
    assert result == inv

def test_cache_get_missing_returns_none():
    cache = InvarianceCache()
    assert cache.get("nonexistent", "a", "b") is None

def test_cache_clear_kernel():
    cache = InvarianceCache()
    cache.put("kernel1", "sig", "cfg1", InvarianceData(...))
    cache.put("kernel1", "sig", "cfg2", InvarianceData(...))
    cache.put("kernel2", "sig", "cfg1", InvarianceData(...))
    cache.clear_kernel("kernel1", "sig")
    assert cache.get("kernel1", "sig", "cfg1") is None
    assert cache.get("kernel2", "sig", "cfg1") is not None

def test_cache_thread_safety():
    """Concurrent put/get from multiple threads."""
    cache = InvarianceCache()
    # Spawn threads, verify no race conditions
```

### Rollback Plan
- Modify `invariance.py` to remove cache code

---

## Phase 4: Storage Layer

### Goal
File-based baseline storage with commit/load/compare operations.

### Inputs
- Phase 2: `InvarianceData`

### Outputs
- `get_baselines_dir()` → Path
- `commit_invariance(kernel_name, sig_hash, inv)` → None (raises on failure)
- `load_baseline(kernel_name, sig_hash)` → Optional[InvarianceData]
- `compare_and_report(kernel_name, sig_hash, current)` → bool (raises on mismatch)
- `NumericalInvarianceMismatchError` exception class

### Files Modified
- `python/triton/runtime/invariance.py`

### Test Strategy
```python
# test_invariance_storage.py
def test_commit_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(invariance, 'get_baselines_dir', lambda: tmp_path)
    inv = InvarianceData(...)
    commit_invariance("my_kernel", "abc123", inv)
    assert (tmp_path / "my_kernel_abc123.json").exists()

def test_load_baseline_returns_data(tmp_path, monkeypatch):
    monkeypatch.setattr(invariance, 'get_baselines_dir', lambda: tmp_path)
    inv = InvarianceData(...)
    commit_invariance("my_kernel", "abc123", inv)
    loaded = load_baseline("my_kernel", "abc123")
    assert loaded == inv

def test_load_baseline_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(invariance, 'get_baselines_dir', lambda: tmp_path)
    assert load_baseline("nonexistent", "xyz") is None

def test_compare_and_report_match_returns_true(tmp_path, monkeypatch):
    # Store baseline, compare with same invariance
    assert compare_and_report("kernel", "sig", inv) is True

def test_compare_and_report_mismatch_raises(tmp_path, monkeypatch):
    # Store baseline, compare with different invariance
    with pytest.raises(NumericalInvarianceMismatchError):
        compare_and_report("kernel", "sig", different_inv)
```

### Rollback Plan
- Modify `invariance.py` to remove storage code

---

## Phase 5: Integration API

### Goal
High-level API functions that compose cache + storage + pybind.

### Inputs
- Phase 1: Pybind binding
- Phase 3: Cache layer
- Phase 4: Storage layer

### Outputs
- `cache_invariance(mod, kernel_name, sig_hash, cfg_hash)` → InvarianceData
- `on_winner_selected(kernel_name, sig_hash, winning_cfg_hash)` → None
- `compute_signature_hash(params)` → str
- `compute_config_hash(config)` → str

### Files Modified
- `python/triton/runtime/invariance.py`

### Test Strategy
```python
# test_invariance_api.py
def test_cache_invariance_calls_pybind_and_caches(mock_mod):
    inv = cache_invariance(mock_mod, "kernel", "sig", "cfg")
    assert isinstance(inv, InvarianceData)
    # Verify it's in cache
    assert _invariance_cache.get("kernel", "sig", "cfg") == inv

def test_cache_invariance_raises_on_failure(broken_mod):
    with pytest.raises(Exception):
        cache_invariance(broken_mod, "kernel", "sig", "cfg")

def test_on_winner_selected_commits_when_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_INVARIANCE_COMMIT", "1")
    # Pre-populate cache
    _invariance_cache.put("kernel", "sig", "cfg", inv)
    on_winner_selected("kernel", "sig", "cfg")
    # Verify file was created
    assert load_baseline("kernel", "sig") == inv

def test_on_winner_selected_compares_when_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_INVARIANCE_COMPARE", "1")
    # Store mismatched baseline
    commit_invariance("kernel", "sig", old_inv)
    _invariance_cache.put("kernel", "sig", "cfg", new_inv)
    with pytest.raises(NumericalInvarianceMismatchError):
        on_winner_selected("kernel", "sig", "cfg")

def test_on_winner_selected_clears_cache():
    _invariance_cache.put("kernel", "sig", "cfg1", inv1)
    _invariance_cache.put("kernel", "sig", "cfg2", inv2)
    on_winner_selected("kernel", "sig", "cfg1")
    assert _invariance_cache.get("kernel", "sig", "cfg1") is None
    assert _invariance_cache.get("kernel", "sig", "cfg2") is None

def test_compute_signature_hash_deterministic():
    params = [...]
    h1 = compute_signature_hash(params)
    h2 = compute_signature_hash(params)
    assert h1 == h2

def test_compute_config_hash_deterministic():
    config = Config(num_warps=4, num_stages=2)
    h1 = compute_config_hash(config)
    h2 = compute_config_hash(config)
    assert h1 == h2
```

### Rollback Plan
- Modify `invariance.py` to remove API functions

---

## Phase 6: make_ttgir Hook

### Goal
Add Phase 1 hook in `compiler.py` to cache invariance after TTGIR generation.

### Inputs
- Phase 5: Integration API

### Outputs
- Modified `third_party/nvidia/backend/compiler.py`
- Invariance cached for each config during compilation

### Hook Location
After `pm.run(mod)` at line ~374 in `make_ttgir`:

```python
# After pm.run(mod) in make_ttgir
if os.environ.get("TRITON_INVARIANCE_COMMIT") or os.environ.get("TRITON_INVARIANCE_COMPARE"):
    from triton.runtime.invariance import cache_invariance, compute_config_hash
    kernel_name = mod.get_entry_func_name()
    sig_hash = compute_signature_hash(metadata.get("signature", {}))
    cfg_hash = compute_config_hash(opt)
    cache_invariance(mod, kernel_name, sig_hash, cfg_hash)
```

### Test Strategy
```python
# test_make_ttgir_hook.py
def test_make_ttgir_caches_invariance(monkeypatch):
    monkeypatch.setenv("TRITON_INVARIANCE_COMMIT", "1")
    
    @triton.jit
    def simple_kernel(x_ptr, y_ptr, n: tl.constexpr):
        tl.store(y_ptr, tl.load(x_ptr))
    
    # Force compilation with specific config
    # Verify invariance is in cache after compilation
    
def test_make_ttgir_no_cache_without_env():
    # Compile without env vars, verify cache is empty
    
def test_make_ttgir_different_configs_different_cache_entries():
    # Compile with two configs, verify two cache entries
```

### Rollback Plan
- Revert `compiler.py` changes (single hunk)

---

## Phase 7: Autotuner Hook

### Goal
Add Phase 2 hook in `autotuner.py` to store winner's invariance.

### Inputs
- Phase 6: make_ttgir hook (ensures cache is populated)

### Outputs
- Modified `python/triton/runtime/autotuner.py`
- Winner's invariance stored to baselines

### Hook Location
After line ~250 where winner is selected:

```python
self.cache[key] = builtins.min(timings, key=timings.get)

# Phase 2 hook: store winner's invariance
if os.environ.get("TRITON_INVARIANCE_COMMIT") or os.environ.get("TRITON_INVARIANCE_COMPARE"):
    from triton.runtime.invariance import on_winner_selected, compute_signature_hash, compute_config_hash
    kernel_name = self.base_fn.__name__
    sig_hash = compute_signature_hash(self.fn.params)
    winning_cfg_hash = compute_config_hash(self.cache[key])
    on_winner_selected(kernel_name, sig_hash, winning_cfg_hash)
```

### Test Strategy
```python
# test_autotuner_hook.py
def test_autotuner_stores_winner_invariance(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_INVARIANCE_COMMIT", "1")
    
    @triton.autotune(configs=[Config(num_warps=4), Config(num_warps=8)], key=['n'])
    @triton.jit
    def tuned_kernel(...):
        ...
    
    # Run autotuning
    tuned_kernel[grid](...)
    
    # Verify baseline file exists with winner's invariance
    
def test_autotuner_compares_against_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_INVARIANCE_COMPARE", "1")
    # Pre-store baseline
    # Run autotuning
    # Verify comparison happened (no error = match)

def test_autotuner_detects_mismatch(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_INVARIANCE_COMPARE", "1")
    # Pre-store DIFFERENT baseline
    # Run autotuning
    # Expect NumericalInvarianceMismatchError
```

### Rollback Plan
- Revert `autotuner.py` changes (single hunk)

---

## Phase 8: E2E Integration

### Goal
Full end-to-end testing with real kernels from `test_tlx.py`.

### Inputs
- All previous phases

### Outputs
- Verified workflow for commit and compare modes
- Documentation updates

### Test Strategy
```python
# test_invariance_e2e.py
def test_e2e_commit_mode(tmp_path, monkeypatch):
    """Full workflow: compile kernel, commit baseline."""
    monkeypatch.setenv("TRITON_INVARIANCE_COMMIT", "1")
    
    # Run a real kernel from test_tlx.py
    from third_party.tlx.tutorials import matmul
    matmul.test_matmul()
    
    # Verify baselines were created
    baselines = list(tmp_path.glob("*.json"))
    assert len(baselines) > 0

def test_e2e_compare_mode_match(tmp_path, monkeypatch):
    """Verify no error when code is unchanged."""
    # First run: commit
    monkeypatch.setenv("TRITON_INVARIANCE_COMMIT", "1")
    run_kernel()
    
    # Second run: compare
    monkeypatch.delenv("TRITON_INVARIANCE_COMMIT")
    monkeypatch.setenv("TRITON_INVARIANCE_COMPARE", "1")
    run_kernel()  # Should succeed without error

def test_e2e_compare_mode_detects_regression():
    """Verify error when numerical behavior changes."""
    # Commit baseline with version A of kernel
    # Modify kernel to version B (different numerical behavior)
    # Compare should raise NumericalInvarianceMismatchError
```

### Rollback Plan
- Full revert of all phases (ordered reverse)

---

## Summary Table

| Phase | Goal | Test File | LOC Est. | Risk | Status |
|-------|------|-----------|----------|------|--------|
| 1 | Pybind binding | `test_pybind_invariance.py` | ~20 | Low | ⬜ |
| 2 | Data layer | `test_invariance_data.py` | ~50 | Low | ⬜ |
| 3 | Cache layer | `test_invariance_cache.py` | ~40 | Low | ⬜ |
| 4 | Storage layer | `test_invariance_storage.py` | ~60 | Low | ⬜ |
| 5 | Integration API | `test_invariance_api.py` | ~80 | Medium | ⬜ |
| 6 | make_ttgir hook | `test_make_ttgir_hook.py` | ~30 | Medium | ⬜ |
| 7 | Autotuner hook | `test_autotuner_hook.py` | ~40 | Medium | ⬜ |
| 8 | E2E integration | `test_invariance_e2e.py` | ~50 | High | ⬜ |

---

## Existing C++ Implementation

The C++ numerical invariance implementation already exists:

- **Header**: `include/triton/Dialect/TritonGPU/Transforms/NumericalInvariance.h`
- **Implementation**: `lib/Dialect/TritonGPU/Transforms/NumericalInvariance.cpp`
- **Unit Tests**: `unittest/Dialect/TritonGPU/NumericalInvarianceTest.cpp` (27 tests)
- **Pass Definitions**: `include/triton/Dialect/TritonGPU/Transforms/Passes.td`

Key API:
```cpp
namespace mlir::triton::gpu {
struct NumericalInvariance {
  size_t computationDagHash;
  std::string dtypeSignature;
  size_t layoutHash;
  size_t hwConfigHash;
  size_t fingerprint() const;
};

NumericalInvariance computeNumericalInvariance(ModuleOp module);
}
```

---

## Storage Format

**Path**: `baselines/numerical_invariance/{kernel_name}_{sig_hash}.json`

```json
{
  "kernel_name": "matmul_kernel",
  "signature_hash": "a1b2c3d4e5f6g7h8",
  "invariance": {
    "fingerprint": 123456789,
    "computation_dag_hash": 987654321,
    "dtype_signature": "input0:f16,input1:f16->output0:f32",
    "layout_hash": 111222333,
    "hw_config_hash": 444555666,
    "config_hash": "cfg_hash_here"
  }
}
```
