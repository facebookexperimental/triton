from __future__ import annotations


_NATIVE_PROFILERS = {
    "amd": "rocprofv3",
    "cuda": "ncu",
    "hip": "rocprofv3",
    "nvidia": "ncu",
    "rocm": "rocprofv3",
}


def native_profiler_for_backend(backend: str) -> str | None:
    return _NATIVE_PROFILERS.get(backend.strip().lower())
