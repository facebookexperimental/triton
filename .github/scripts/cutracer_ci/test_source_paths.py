from pathlib import Path

from sample_logs import SYNCCHECK_ERROR
from sanitizer_log import device_reports
from source_paths import compiled_kernels, resolve_paths


def _cache_ir(cache: Path, key: str, kernel: str, *paths: str) -> None:
    (cache / key).mkdir(parents=True)
    locs = "\n".join(f'#loc{i} = loc("{path}":{i}:0)' for i, path in enumerate(paths))
    (cache / key / f"{kernel}.ttgir").write_text(locs)


def test_resolve_paths_from_cached_ir(tmp_path):
    site = "/venv/lib/python3.12/site-packages/triton/tlx/ops/kernels"
    _cache_ir(tmp_path, "A", "_attn_fwd_ws", f"{site}/flash_attn/sm100.py", f"{site}/flash_attn/common.py")
    _cache_ir(tmp_path, "B", "_gemm_ws", f"{site}/mm/sm100.py")
    _cache_ir(tmp_path, "C", "_ambiguous", "/a/sm100.py", "/b/sm100.py")
    reports = device_reports(SYNCCHECK_ERROR)
    reports += [{"header": "x", "kernel": kernel, "file": file, "line": 1, "path": None}
                for kernel, file in [("_gemm_ws", "sm100.py"), ("_ambiguous", "sm100.py"), ("not_cached", "k.py")]]
    resolve_paths(reports, tmp_path)
    assert [r["path"] for r in reports] == [
        "triton/tlx/ops/kernels/flash_attn/sm100.py",
        "triton/tlx/ops/kernels/mm/sm100.py",
        None,
        None,
    ]


def test_compiled_kernels_lists_each_kernel_once(tmp_path):
    _cache_ir(tmp_path, "A", "_attn_fwd_ws", "/x/sm100.py")
    _cache_ir(tmp_path, "B", "_attn_fwd_ws", "/x/sm100.py")
    _cache_ir(tmp_path, "C", "_gemm_ws", "/y/sm100.py")
    (tmp_path / "C" / "_gemm_ws.json").write_text("{}")
    assert compiled_kernels(tmp_path) == ["_attn_fwd_ws", "_gemm_ws"]
