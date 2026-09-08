from __future__ import annotations


_NATIVE_PROFILERS = {
    "cuda": "ncu",
    "nvidia": "ncu",
}


def native_profiler_for_backend(backend: str) -> str | None:
    return _NATIVE_PROFILERS.get(backend.strip().lower())
