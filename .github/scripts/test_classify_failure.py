#!/usr/bin/env python3
"""Unit tests for classify_failure.classify (stdlib-only; run with pytest)."""

import classify_failure as cf


def test_gpu_busy_is_external():
    ext, reason = cf.classify("RuntimeError: all CUDA-capable devices are busy or unavailable")
    assert ext is True
    assert "GPU" in reason


def test_no_cuda_device_is_external():
    ext, _ = cf.classify("No CUDA GPUs are available")
    assert ext is True


def test_driver_mismatch_is_external():
    ext, _ = cf.classify("CUDA driver version is insufficient for CUDA runtime version")
    assert ext is True


def test_network_proxy_is_external():
    ext, reason = cf.classify("Received HTTP code 407 Proxy Authentication Required from proxy")
    assert ext is True
    assert "roxy" in reason


def test_llvm_download_is_external():
    ext, _ = cf.classify("Failed to download LLVM tarball from blob storage")
    assert ext is True


def test_disk_full_is_external():
    ext, _ = cf.classify("OSError: [Errno 28] No space left on device")
    assert ext is True


def test_out_of_resources_is_not_external():
    # Compiler resource regression -> must be chased by bisection, not excused.
    ext, reason = cf.classify("OutOfResources: shared memory, Required: 232712, Hardware limit: 232448")
    assert ext is False
    assert reason == ""


def test_assertion_is_not_external():
    ext, _ = cf.classify("AssertionError: Tensor-likes are not close! max abs diff 3.4")
    assert ext is False


def test_not_external_wins_over_infra_lookalike():
    # Even if an infra-ish word appears, a concrete correctness error dominates.
    ext, _ = cf.classify("AssertionError: out of memory expectation mismatch in test")
    assert ext is False


def test_empty_summary():
    assert cf.classify("") == (False, "")
    assert cf.classify(None) == (False, "")


def test_generic_test_failure_is_not_external():
    ext, _ = cf.classify("ValueError: shapes do not match")
    assert ext is False
