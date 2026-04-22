"""Unit tests for _LRUDict used in kernel and fast-path caches.

These tests are pure-Python and do not require a GPU.
"""

import threading

from triton.runtime.jit import _LRUDict  # type: ignore[import-not-found]


class TestLRUDict:

    def test_basic_get_put(self):
        d = _LRUDict(3)
        d.put("a", 1)
        d.put("b", 2)
        d.put("c", 3)
        assert len(d) == 3
        assert d.get("a") == 1
        assert d.get("b") == 2
        assert d.get("c") == 3

    def test_miss_returns_none(self):
        d = _LRUDict(3)
        assert d.get("missing") is None

    def test_eviction_respects_maxsize(self):
        """After 256 unique keys, cache size stays at 256."""
        maxsize = 256
        d = _LRUDict(maxsize)
        for i in range(1000):
            d.put(f"key_{i}", i)
            assert len(d) <= maxsize
        assert len(d) == maxsize

    def test_lru_eviction_order(self):
        d = _LRUDict(3)
        d.put("a", 1)
        d.put("b", 2)
        d.put("c", 3)
        # 'd' evicts 'a' (least recently used)
        d.put("d", 4)
        assert len(d) == 3
        assert d.get("a") is None
        assert d.get("b") == 2
        assert d.get("c") == 3
        assert d.get("d") == 4

    def test_access_refreshes_lru_position(self):
        d = _LRUDict(3)
        d.put("a", 1)
        d.put("b", 2)
        d.put("c", 3)
        # Access 'a' — moves it to most-recently-used
        d.get("a")
        # Insert 'd' — should evict 'b' (now LRU), not 'a'
        d.put("d", 4)
        assert d.get("a") == 1
        assert d.get("b") is None
        assert d.get("c") == 3
        assert d.get("d") == 4

    def test_put_existing_key_updates_value(self):
        d = _LRUDict(3)
        d.put("a", 1)
        d.put("b", 2)
        d.put("c", 3)
        d.put("a", 10)
        assert len(d) == 3
        assert d.get("a") == 10

    def test_maxsize_zero_disables_cache(self):
        d = _LRUDict(0)
        d.put("a", 1)
        assert len(d) == 0
        assert d.get("a") is None

    def test_clear(self):
        d = _LRUDict(10)
        d.put("a", 1)
        d.put("b", 2)
        d.clear()
        assert len(d) == 0
        assert d.get("a") is None
        assert d.get("b") is None

    def test_bounded_after_many_unique_shapes(self):
        """Simulates vLLM-like workload with many unique shapes.

        Verifies memory is bounded: only maxsize entries retained.
        """
        maxsize = 32
        d = _LRUDict(maxsize)
        # Simulate 500 unique shape specializations
        for i in range(500):
            key = (0, ("float32", i, 16))  # mimics fast_key structure
            d.put(key, (f"kernel_{i}", f"metadata_{i}"))
        assert len(d) == maxsize
        # Only the last `maxsize` entries should be present
        for i in range(500 - maxsize, 500):
            key = (0, ("float32", i, 16))
            assert d.get(key) is not None, f"Expected key for shape {i} to be present"
        # Earlier entries should be evicted
        for i in range(500 - maxsize):
            key = (0, ("float32", i, 16))
            assert d.get(key) is None, f"Expected key for shape {i} to be evicted"

    def test_concurrent_put_and_get(self):
        """Stress-test thread safety: concurrent puts/gets must not corrupt state."""
        maxsize = 64
        d = _LRUDict(maxsize)
        errors = []
        barrier = threading.Barrier(4)

        def writer(start):
            try:
                barrier.wait()
                for i in range(start, start + 500):
                    d.put(f"key_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader(start):
            try:
                barrier.wait()
                for i in range(start, start + 500):
                    d.get(f"key_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0, )),
            threading.Thread(target=writer, args=(500, )),
            threading.Thread(target=reader, args=(0, )),
            threading.Thread(target=reader, args=(500, )),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Concurrent access raised exceptions: {errors}"
        assert len(d) <= maxsize

    def test_deepcopy(self):
        """_LRUDict must survive deepcopy (used by _DeviceCaches.__deepcopy__)."""
        import copy
        d = _LRUDict(3)
        d.put("a", 1)
        d.put("b", 2)
        d2 = copy.deepcopy(d)
        assert len(d2) == 2
        assert d2.get("a") == 1
        assert d2.get("b") == 2
        # Mutating the copy must not affect the original.
        d2.put("c", 3)
        d2.put("d", 4)  # evicts "a" in d2
        assert d2.get("a") is None
        assert d.get("a") == 1  # original untouched

    def test_on_evict_callback(self):
        """on_evict is called with (key, value) when an entry is evicted."""
        evicted = []
        d = _LRUDict(3, on_evict=lambda k, v: evicted.append((k, v)))
        d.put("a", 1)
        d.put("b", 2)
        d.put("c", 3)
        assert evicted == []
        # 'd' evicts 'a'
        d.put("d", 4)
        assert evicted == [("a", 1)]
        # 'e' evicts 'b'
        d.put("e", 5)
        assert evicted == [("a", 1), ("b", 2)]

    def test_on_evict_called_on_clear(self):
        """on_evict is called for all entries when clear() is called."""
        evicted = []
        d = _LRUDict(3, on_evict=lambda k, v: evicted.append((k, v)))
        d.put("a", 1)
        d.put("b", 2)
        d.clear()
        assert len(evicted) == 2
        assert ("a", 1) in evicted
        assert ("b", 2) in evicted
