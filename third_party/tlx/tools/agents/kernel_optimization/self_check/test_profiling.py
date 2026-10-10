from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock
from unittest.mock import patch

from ..decision_maker.profiling import rocm_profiler as rocm_profiler_module
from ..decision_maker.profiling.amd_att import (
    collect_amd_att,
    compare_amd_att_profiles,
)
from ..decision_maker.profiling.amdgcn_isa import analyze_amdgcn, collect_amdgcn_isa
from ..decision_maker.profiling import (
    ProfileRequest,
    compact_profile_summary,
    export_ncu_report_details,
    extract_native_profiler_duration_us,
    extract_ncu_duration_us,
    native_profiler_regression_diagnostic,
    ncu_regression_diagnostic,
    normalize_ncu_metrics,
    normalize_profile_request,
    parse_ncu_csv,
    parse_ncu_query_metrics,
    parse_proton_launch_attribution,
    per_case_profile_request,
    resolve_profile_request_for_target,
    resolve_profile_tools,
    select_ncu_metric_names,
)
from ..decision_maker.profiling.rocm_profiler import (
    collect_rocprofv3,
    filter_extreme_timing_outliers,
    find_rocprofv3,
    parse_rocprof_counter_collection,
    parse_rocprof_kernel_trace,
)


class ProfileRequestTest(unittest.TestCase):
    def test_bool_and_mapping_normalization(self) -> None:
        self.assertIsNone(normalize_profile_request(False))
        request = normalize_profile_request(True)
        self.assertIsInstance(request, ProfileRequest)
        assert request is not None
        self.assertEqual(request.level, "summary")

        with tempfile.TemporaryDirectory() as directory:
            mapped = normalize_profile_request(
                {
                    "level": "deep",
                    "tools": ["native_profiler"],
                    "experiment_id": "baseline",
                    "artifacts_dir": directory,
                    "reason": "baseline profile",
                }
            )
            assert mapped is not None
            self.assertEqual(mapped.level, "deep")
            self.assertEqual(mapped.tools, ("native_profiler",))
            self.assertEqual(mapped.artifacts_dir, Path(directory))

    def test_validates_request_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "summary.*deep"):
            ProfileRequest(level="diagnostic")
        with self.assertRaisesRegex(ValueError, "absolute"):
            ProfileRequest(artifacts_dir=Path("relative"))
        with self.assertRaisesRegex(ValueError, "diagnostic_only"):
            ProfileRequest(tools=("proton_intra_kernel",))
        with self.assertRaisesRegex(ValueError, "warp"):
            ProfileRequest(
                tools=("proton_intra_kernel",),
                diagnostic_only=True,
                granularity="warp_group",
            )
        request = ProfileRequest(
            tools=("proton_intra_kernel",),
            diagnostic_only=True,
            granularity="warp",
        )
        self.assertEqual(request.granularity, "warp")

    def test_resolves_native_profiler_and_deduplicates_aliases(self) -> None:
        self.assertEqual(
            resolve_profile_tools(
                ("proton_launch", "native_profiler", "ncu"),
                native_profiler="ncu",
            ),
            ("proton_launch", "ncu"),
        )
        self.assertEqual(
            resolve_profile_tools((), native_profiler="ncu", default=("ncu",)),
            ("ncu",),
        )
        self.assertEqual(
            resolve_profile_tools(("native_profiler",)),
            ("native_profiler",),
        )

    def test_resolves_native_profiler_for_target_backend(self) -> None:
        payload = ProfileRequest(
            tools=("proton_launch", "native_profiler", "ncu"),
        ).to_json()
        cuda = resolve_profile_request_for_target(payload, {"backend": "cuda"})
        assert cuda is not None
        self.assertEqual(cuda["tools"], ["proton_launch", "ncu"])

        amd = resolve_profile_request_for_target(payload, {"backend": "hip"})
        assert amd is not None
        self.assertEqual(
            amd["tools"],
            ["rocprofv3", "ncu", "amdgcn_isa"],
        )

        hip = resolve_profile_request_for_target(payload, {"backend": "hip"})
        assert hip is not None
        self.assertEqual(hip["tools"], ["rocprofv3", "ncu", "amdgcn_isa"])

        deep = resolve_profile_request_for_target(
            ProfileRequest(
                level="deep",
                tools=("proton_launch", "native_profiler"),
            ).to_json(),
            {"backend": "hip"},
        )
        assert deep is not None
        self.assertEqual(deep["tools"], ["amd_att", "rocprofv3", "amdgcn_isa"])
        self.assertNotIn("amdgcn_isa", cuda["tools"])

    def test_per_case_profile_request_expands_absolute_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = ProfileRequest(artifacts_dir=Path(directory)).to_json()
            request = per_case_profile_request(payload, "shape/128?case")
            assert request is not None
            artifacts_dir = Path(str(request["artifacts_dir"]))
            self.assertTrue(artifacts_dir.is_absolute())
            self.assertTrue(artifacts_dir.exists())
            self.assertEqual(artifacts_dir.parent, Path(directory))
            self.assertNotIn("/", artifacts_dir.name)


class ProfileParsingTest(unittest.TestCase):
    def test_parse_proton_tree_into_launch_attribution_summary(self) -> None:
        proton = {
            "name": "root",
            "children": [
                {"name": "python wrapper", "time_us": 5.0, "count": 2},
                {"name": "main kernel launch", "duration_ns": 7000, "count": 1},
                {"name": "helper kernel", "duration_us": 3.0, "count": 1},
                {
                    "name": "nested",
                    "children": [{"name": "dispatch wrapper", "time_ms": 0.002}],
                },
            ],
        }
        summary = parse_proton_launch_attribution(proton)
        self.assertEqual(summary["schema"], "launch_attribution_only")
        totals = summary["totals"]
        self.assertAlmostEqual(totals["wrapper_us"], 7.0)
        self.assertAlmostEqual(totals["main_kernel_us"], 7.0)
        self.assertAlmostEqual(totals["non_main_kernel_us"], 3.0)
        self.assertEqual(totals["count"], 5)

    def test_parse_proton_hatchet_tree_with_explicit_main_scope(self) -> None:
        proton = [
            {
                "children": [
                    {
                        "children": [],
                        "frame": {
                            "name": "_attn_bwd_preprocess",
                            "type": "function",
                        },
                        "metrics": {
                            "count": 2,
                            "device_id": "0",
                            "device_type": "CUDA",
                            "time (ns)": 160448,
                        },
                    },
                    {
                        "children": [],
                        "frame": {
                            "name": "_attn_bwd_mxf8_ws",
                            "type": "function",
                        },
                        "metrics": {
                            "count": 3,
                            "device_id": "0",
                            "device_type": "CUDA",
                            "time (ns)": 9322456,
                        },
                    },
                ],
                "frame": {"name": "ROOT", "type": "function"},
                "metrics": {"count": 0, "time (ns)": 0},
            },
            {"CUDA": {"0": {"arch": "100"}}},
        ]

        summary = parse_proton_launch_attribution(
            proton, main_scope="_attn_bwd_mxf8_ws"
        )

        self.assertEqual(
            summary["leaves"],
            [
                {
                    "name": "_attn_bwd_preprocess",
                    "time_us": 160.448,
                    "count": 2,
                },
                {"name": "_attn_bwd_mxf8_ws", "time_us": 9322.456, "count": 3},
            ],
        )
        totals = summary["totals"]
        self.assertAlmostEqual(totals["leaf_time_us"], 9482.904)
        self.assertAlmostEqual(totals["main_kernel_us"], 9322.456)
        self.assertAlmostEqual(totals["non_main_kernel_us"], 160.448)
        self.assertEqual(totals["wrapper_us"], 0.0)
        self.assertEqual(totals["count"], 5)

    def test_parse_ncu_csv_and_metric_selection(self) -> None:
        csv_text = (
            "ID,Metric Name,Metric Unit,Metric Value\n"
            '0,gpu__time_duration.sum,ns,"2,000"\n'
            "1,sm__throughput.avg.pct_of_peak_sustained_elapsed,%,75.5\n"
        )
        metrics = parse_ncu_csv(csv_text)
        self.assertEqual(
            metrics["gpu__time_duration.sum"], {"value": 2000.0, "unit": "ns"}
        )
        selected = select_ncu_metric_names(metrics.keys(), "summary")
        self.assertEqual(selected["metrics"]["duration_us"], "gpu__time_duration.sum")
        self.assertIsNone(selected["metrics"]["dram_throughput_pct"])
        self.assertTrue(selected["diagnostics"])

        normalized = normalize_ncu_metrics(metrics, "summary")
        self.assertAlmostEqual(normalized["summary"]["duration_us"], 2.0)
        self.assertAlmostEqual(normalized["summary"]["sm_throughput_pct"], 75.5)
        self.assertIsNone(normalized["summary"]["dram_throughput_pct"])

    def test_normalizes_local_memory_load_bytes(self) -> None:
        metric = "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"
        selected = select_ncu_metric_names((metric,), "deep")
        self.assertEqual(selected["metrics"]["local_load_bytes"], metric)

        normalized = normalize_ncu_metrics(
            {metric: {"value": 4096.0, "unit": "byte"}},
            "deep",
        )
        self.assertEqual(normalized["registers"]["local_load_bytes"], 4096.0)

    def test_exports_and_normalizes_ncu_report_details(self) -> None:
        details_csv = (
            'ID,Metric Name,Metric Unit,Metric Value\n'
            '0,gpu__time_duration.sum,ms,17.18\n'
            '0,sm__throughput.avg.pct_of_peak_sustained_elapsed,%,39.83\n'
            '0,dram__bytes_read.sum,Gbyte,1.52\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "profile.ncu-rep"
            report.write_bytes(b"report")
            artifacts_dir = root / "export"
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=details_csv, stderr="warning\n"
            )
            with mock.patch("subprocess.run", return_value=completed) as run:
                result = export_ncu_report_details(
                    report,
                    artifacts_dir=artifacts_dir,
                    level="deep",
                    ncu_binary="/opt/ncu",
                )

            run.assert_called_once_with(
                [
                    "/opt/ncu",
                    "--import",
                    str(report),
                    "--page",
                    "details",
                    "--csv",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=120.0,
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["ncu"]["summary"]["duration_us"], 17180.0)
            self.assertEqual(
                result["ncu"]["summary"]["sm_throughput_pct"], 39.83
            )
            self.assertEqual(
                result["ncu"]["memory"]["dram_read_bytes"],
                1_520_000_000.0,
            )
            artifacts = result["artifacts"]
            self.assertTrue(all(Path(path).is_absolute() for path in artifacts.values()))
            self.assertEqual(Path(artifacts["ncu_details_csv"]).read_text(), details_csv)
            self.assertEqual(
                Path(artifacts["ncu_import_stderr"]).read_text(), "warning\n"
            )

    def test_preserves_failed_ncu_report_import_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "profile.ncu-rep"
            report.write_bytes(b"report")
            completed = subprocess.CompletedProcess(
                args=[], returncode=2, stdout="status\n", stderr="bad metric\n"
            )
            with mock.patch("subprocess.run", return_value=completed):
                result = export_ncu_report_details(
                    report,
                    artifacts_dir=root / "export",
                )

            self.assertFalse(result["success"])
            self.assertTrue(
                any("exit code 2" in item for item in result["diagnostics"])
            )
            self.assertEqual(
                Path(result["artifacts"]["ncu_details_csv"]).read_text(),
                "status\n",
            )
            self.assertEqual(
                Path(result["artifacts"]["ncu_import_stderr"]).read_text(),
                "bad metric\n",
            )
            self.assertIsNone(result["ncu"]["summary"]["duration_us"])

    def test_handles_missing_and_timed_out_ncu_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.ncu-rep"
            with mock.patch("subprocess.run") as run:
                result = export_ncu_report_details(
                    missing,
                    artifacts_dir=root / "missing-export",
                )
            run.assert_not_called()
            self.assertFalse(result["success"])
            self.assertTrue(
                any("does not exist" in item for item in result["diagnostics"])
            )

            report = root / "profile.ncu-rep"
            report.write_bytes(b"report")
            timeout = subprocess.TimeoutExpired(
                cmd=["ncu"], timeout=0.1, output=b"partial", stderr=b"timeout"
            )
            with mock.patch("subprocess.run", side_effect=timeout):
                result = export_ncu_report_details(
                    report,
                    artifacts_dir=root / "timeout-export",
                    timeout_s=0.1,
                )
            self.assertFalse(result["success"])
            self.assertTrue(
                any("timed out" in item for item in result["diagnostics"])
            )
            self.assertEqual(
                Path(result["artifacts"]["ncu_details_csv"]).read_text(),
                "partial",
            )
            self.assertEqual(
                Path(result["artifacts"]["ncu_import_stderr"]).read_text(),
                "timeout",
            )

    def test_rejects_relative_ncu_export_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "report path must be absolute"):
            export_ncu_report_details(
                "profile.ncu-rep", artifacts_dir="/tmp/ncu-export"
            )
        with self.assertRaisesRegex(ValueError, "artifacts directory must be absolute"):
            export_ncu_report_details(
                "/tmp/profile.ncu-rep", artifacts_dir="relative"
            )

    def test_parse_ncu_query_metrics_accepts_csv_and_plain_text(self) -> None:
        csv_metrics = parse_ncu_query_metrics(
            "Metric Name,Description\n"
            "gpu__time_duration.sum,Duration\n"
            "sm__throughput.avg.pct_of_peak_sustained_elapsed,SM\n"
        )
        self.assertIn("gpu__time_duration.sum", csv_metrics)
        text_metrics = parse_ncu_query_metrics(
            "gpu__time_duration.sum\n"
            "sm__throughput.avg.pct_of_peak_sustained_elapsed some description\n"
        )
        self.assertIn("gpu__time_duration.sum", text_metrics)

    def test_parse_ncu_query_metrics_skips_command_preamble(self) -> None:
        metrics = parse_ncu_query_metrics(
            "$ /usr/local/cuda-12.8/bin/ncu --query-metrics --csv\n"
            "\n"
            '"Metric name","Metric type","Metric unit",'
            '"Metric description","NVIDIA B200"\n'
            '"gpu__time_duration","Counter","ns","Duration","1"\n'
            '"sm__throughput","Throughput","%","SM throughput","1"\n'
        )

        self.assertEqual(metrics, {"gpu__time_duration", "sm__throughput"})

    def test_selects_b200_base_and_scoped_metric_names(self) -> None:
        supported = {
            "gpu__time_duration",
            "sm__throughput",
            "TriageCompute.sm__throughput",
            "FBSP.TriageCompute.dram__throughput",
            "smsp__warp_issue_stalled_barrier_per_warp_active",
            "l1tex__t_bytes_pipe_lsu_mem_local_op_ld",
            "dram__bytes_read",
        }

        selected = select_ncu_metric_names(supported, "deep")

        self.assertEqual(
            selected["metrics"]["duration_us"], "gpu__time_duration.sum"
        )
        self.assertEqual(
            selected["metrics"]["sm_throughput_pct"],
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        )
        self.assertEqual(
            selected["metrics"]["dram_throughput_pct"],
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        )
        self.assertEqual(
            selected["metrics"]["barrier_pct"],
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        )
        self.assertEqual(
            selected["metrics"]["local_load_bytes"],
            "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
        )
        self.assertEqual(
            selected["metrics"]["dram_read_bytes"], "dram__bytes_read.sum"
        )
        self.assertIsNone(selected["metrics"]["registers_per_thread"])
        self.assertIsNone(selected["metrics"]["tensor_activity_pct"])

    def test_normalizes_b200_base_metrics(self) -> None:
        normalized = normalize_ncu_metrics(
            {
                "gpu__time_duration": {"value": 125000.0, "unit": "ns"},
                "sm__throughput": {"value": 75.0, "unit": "%"},
                "dram__throughput": {"value": 50.0, "unit": "%"},
                "dram__bytes_read": {"value": 1.52, "unit": "Gbyte"},
                "smsp__warp_issue_stalled_wait_per_warp_active": {
                    "value": 12.5,
                    "unit": "%",
                },
            },
            "deep",
        )

        self.assertEqual(normalized["summary"]["duration_us"], 125.0)
        self.assertEqual(normalized["summary"]["sm_throughput_pct"], 75.0)
        self.assertEqual(normalized["summary"]["dram_throughput_pct"], 50.0)
        self.assertEqual(
            normalized["memory"]["dram_read_bytes"], 1_520_000_000.0
        )
        self.assertEqual(normalized["stalls"]["async_wait_pct"], 12.5)
        self.assertIsNone(normalized["compute"]["tensor_activity_pct"])

    def test_does_not_match_ncu_sibling_derivatives(self) -> None:
        selected = select_ncu_metric_names(
            {
                "sm__throughput.avg.pct_of_peak_sustained_active",
                "dram__bytes_read.avg",
            },
            "deep",
        )
        self.assertIsNone(selected["metrics"]["sm_throughput_pct"])
        self.assertIsNone(selected["metrics"]["dram_read_bytes"])

        normalized = normalize_ncu_metrics(
            {
                "gpu__time_duration.max": {"value": 250000.0, "unit": "ns"},
                "gpu__time_duration.min": {"value": 125000.0, "unit": "ns"},
            },
            "summary",
        )
        self.assertIsNone(normalized["summary"]["duration_us"])
        self.assertTrue(
            any(
                "missing NCU metric for duration_us" in diagnostic
                for diagnostic in normalized["diagnostics"]
            )
        )

    def test_duration_extraction_and_regression_diagnostic(self) -> None:
        old_flat = {"ncu": {"gpu__time_duration.sum": {"value": 1000, "unit": "ns"}}}
        normalized = {"ncu": {"summary": {"duration_us": 1.02}}}
        self.assertAlmostEqual(extract_ncu_duration_us(old_flat), 1.0)
        self.assertAlmostEqual(extract_ncu_duration_us(normalized), 1.02)
        self.assertIn("regressed", ncu_regression_diagnostic(old_flat, normalized))
        self.assertEqual(ncu_regression_diagnostic(normalized, old_flat), "")

        rocprof_baseline = {"rocprofv3": {"summary": {"duration_us": 10.0}}}
        rocprof_candidate = {"rocprofv3": {"summary": {"duration_us": 10.6}}}
        self.assertEqual(
            extract_native_profiler_duration_us(rocprof_baseline),
            ("rocprofv3", 10.0),
        )
        self.assertIn(
            "rocprofv3 duration regressed",
            native_profiler_regression_diagnostic(rocprof_baseline, rocprof_candidate),
        )
        self.assertEqual(
            native_profiler_regression_diagnostic(old_flat, rocprof_candidate),
            "",
        )
        self.assertEqual(
            native_profiler_regression_diagnostic(
                rocprof_baseline,
                {"rocprofv3": {"summary": {"duration_us": 10.2}}},
            ),
            "",
        )
        self.assertEqual(
            extract_native_profiler_duration_us(
                {"rocprofv3": {"summary": {"duration_us": "invalid"}}}
            ),
            (None, None),
        )

    def test_parses_rocprof_kernel_trace_and_resources(self) -> None:
        trace = (
            "Kernel_Name,Start_Timestamp,End_Timestamp,Workgroup_Size,"
            "LDS_Block_Size,VGPR_Count,Accum_VGPR_Count,SGPR_Count\n"
            "helper,0,1000,64,0,8,0,16\n"
            "gemm,1000,5000,256,32768,64,16,32\n"
            "gemm,5000,9200,256,32768,64,16,32\n"
            "gemm,9200,13000,256,32768,64,16,32\n"
        )
        profile = parse_rocprof_kernel_trace(trace, sample_count=2)
        self.assertEqual(profile["summary"]["dominant_kernel"], "gemm")
        self.assertEqual(profile["summary"]["duration_us"], 4.0)
        self.assertEqual(profile["summary"]["scope"], "dominant_kernel")
        self.assertEqual(profile["kernels"][0]["dispatches"], 3)
        self.assertEqual(profile["kernels"][0]["resources"]["lds_bytes"], 32768.0)

    def test_parses_rocprof_counter_collection(self) -> None:
        counters = (
            "Kernel_Name,Counter_Name,Counter_Value\n"
            "gemm,SQ_WAVES,100\n"
            "gemm,SQ_WAVES,120\n"
            "gemm,SQ_INSTS_MFMA,40\n"
            "helper,SQ_WAVES,2\n"
        )
        parsed = parse_rocprof_counter_collection(counters)
        self.assertEqual(parsed["gemm"]["SQ_WAVES"], 110.0)
        self.assertEqual(parsed["gemm"]["SQ_INSTS_MFMA"], 40.0)
        self.assertEqual(parsed["helper"]["SQ_WAVES"], 2.0)

    def test_filters_only_extreme_rocprof_timing_outliers(self) -> None:
        samples = [100.0, 101.0, 99.0, 102.0, 98.0, 1000.0]

        self.assertEqual(
            filter_extreme_timing_outliers(samples),
            [100.0, 101.0, 99.0, 102.0, 98.0],
        )
        self.assertEqual(
            filter_extreme_timing_outliers([100.0, 100.0, 100.0, 100.0]),
            [100.0, 100.0, 100.0, 100.0],
        )

    def test_finds_newest_internal_rocprof_by_numeric_version(self) -> None:
        candidates = (
            Path("/usr/local/fbcode/platform010/lib/rocm-6.9/bin/rocprofv3"),
            Path("/usr/local/fbcode/platform010/lib/rocm-6.10/bin/rocprofv3"),
        )

        with (
            patch.object(
                rocm_profiler_module.shutil,
                "which",
                return_value=None,
            ),
            patch.object(Path, "is_dir", autospec=True, return_value=True),
            patch.object(Path, "glob", autospec=True, return_value=iter(candidates)),
            patch.object(
                Path,
                "is_file",
                autospec=True,
                side_effect=lambda path: path in candidates,
            ),
            patch.object(
                rocm_profiler_module.os,
                "access",
                return_value=True,
            ),
        ):
            profiler = find_rocprofv3({"PATH": ""})

        self.assertEqual(profiler, candidates[1])

    def test_collects_rocprof_trace_and_counters_from_external_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiler = root / "rocprofv3"
            profiler.write_text(
                "#!/usr/bin/python3\n"
                "import os\n"
                "import sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "assert os.environ['LD_PRELOAD'] == '/usr/lib64/libzstd.so.1'\n"
                "assert args[args.index('--mangled-kernels') + 1] == 'true'\n"
                "output_dir = Path(args[args.index('--output-directory') + 1])\n"
                "output_file = args[args.index('--output-file') + 1]\n"
                "output_dir.mkdir(parents=True, exist_ok=True)\n"
                "(Path.cwd() / '.rocprofv3').mkdir()\n"
                "if '--kernel-trace' in args:\n"
                "    (output_dir / f'{output_file}_kernel_trace.csv').write_text(\n"
                "        'Kernel_Name,Start_Timestamp,End_Timestamp,VGPR_Count\\n'\n"
                "        'gemm,0,4000,64\\n'\n"
                "        'gemm,4000,8200,64\\n'\n"
                "        'gemm,8200,12000,64\\n'\n"
                "    )\n"
                "else:\n"
                "    counters = args[args.index('--pmc') + 1].split(',')\n"
                "    rows = ''.join(f'gemm,{counter},100\\n' for counter in counters)\n"
                "    (output_dir / f'{output_file}_counter_collection.csv').write_text(\n"
                "        'Kernel_Name,Counter_Name,Counter_Value\\n' + rows\n"
                "    )\n"
                "print('fake rocprofv3')\n"
            )
            profiler.chmod(0o755)

            profile = collect_rocprofv3(
                ("/usr/bin/python3", "-c", "pass"),
                root / "artifacts",
                environment={
                    "TLX_ROCPROFV3": str(profiler),
                    "LD_PRELOAD": (
                        "/tmp/librocprofiler-sdk-tool.so:/usr/lib64/libzstd.so.1"
                    ),
                },
            )

            self.assertEqual(profile["tool"], "rocprofv3")
            self.assertEqual(profile["summary"]["dominant_kernel"], "gemm")
            self.assertEqual(profile["summary"]["duration_us"], 4.0)
            self.assertEqual(profile["counters"]["MfmaUtil"], 100.0)
            self.assertEqual(profile["kernels"][0]["samples_us"], [4.0, 4.2, 3.8])
            self.assertTrue(Path(profile["artifacts"]["kernel_trace_csv"]).exists())
            self.assertTrue(
                (
                    Path(profile["artifacts"]["kernel_trace_csv"]).parent / ".rocprofv3"
                ).is_dir()
            )

    def test_collect_rocprof_returns_error_for_unreadable_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiler = root / "rocprofv3"
            profiler.write_text(
                "#!/usr/bin/python3\n"
                "import sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "output_dir = Path(args[args.index('--output-directory') + 1])\n"
                "output_file = args[args.index('--output-file') + 1]\n"
                "output_dir.mkdir(parents=True, exist_ok=True)\n"
                "(output_dir / f'{output_file}_kernel_trace.csv').write_bytes(b'\\xff')\n"
            )
            profiler.chmod(0o755)

            profile = collect_rocprofv3(
                ("/usr/bin/python3", "-c", "pass"),
                root / "artifacts",
                level="timing",
                environment={"TLX_ROCPROFV3": str(profiler)},
            )

            self.assertIn("failed to read rocprofv3 kernel trace", profile["error"])
            self.assertTrue(Path(profile["artifacts"]["command"]).exists())

    def test_gfx950_benchmark_surfaces_rocprof_error(self) -> None:
        harness_path = (
            Path(__file__).parents[1]
            / "decision_maker"
            / "targets"
            / "amd"
            / "gfx950"
            / "gemm"
            / "harness.py"
        )
        spec = importlib.util.spec_from_file_location(
            "test_gfx950_harness", harness_path
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertIsNotNone(spec.loader)
        assert spec.loader is not None
        harness = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": ModuleType("torch")}):
            spec.loader.exec_module(harness)

        with (
            patch.object(
                harness,
                "collect_rocprofv3",
                return_value={
                    "error": "rocprofv3 was not found; set TLX_ROCPROFV3 or PATH"
                },
            ),
            self.assertRaisesRegex(RuntimeError, "set TLX_ROCPROFV3 or PATH"),
        ):
            harness.benchmark(
                (None, None, Path("/tmp/candidate.py"), "cuda"),
                {"case_id": "square"},
                repetitions=3,
            )

    def test_production_mm_profile_invokes_native_profiler_for_selected_winner(self) -> None:
        from ..decision_maker.tuning_harnesses import mm

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidate.py"
            source.write_text("def mm():\n    pass\n")
            artifact = {
                "records": {
                    "shape": {
                        "full_best_config": "kernel: BLOCK_M=64 num_warps=4 num_stages=2",
                        "full_config_timings": [],
                    }
                },
                "source_path": source,
                "device": "cuda",
                "arch": "gfx942",
            }
            expected = {
                "tool": "rocprofv3",
                "summary": {"duration_us": 12.0, "dominant_kernel": "gemm"},
            }
            with patch.object(mm, "collect_rocprofv3", return_value=expected) as collect:
                profile = mm.profile(
                    artifact,
                    {"case_id": "shape", "parameters": {}},
                    {
                        "tools": ["rocprofv3"],
                        "level": "deep",
                        "artifacts_dir": str(root / "artifacts"),
                    },
                )
            self.assertEqual(profile["rocprofv3"], expected)
            workload = collect.call_args.args[0]
            self.assertIn("--winner", workload)
            self.assertIn("gfx942", workload)

    def test_collects_and_validates_rocprofv3_att_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiler = root / "rocprofv3"
            profiler.write_text(
                "#!/usr/bin/python3\n"
                "import sqlite3, sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "if '--help' in args:\n"
                "    print('--att --rocm-root')\n"
                "    raise SystemExit(0)\n"
                "assert '--' in args\n"
                "assert args[args.index('--att-activity') + 1] == '8'\n"
                "output_dir = Path(args[args.index('-d') + 1])\n"
                "output_dir.mkdir(parents=True)\n"
                "db = sqlite3.connect(output_dir / 'tlx_att_results.db')\n"
                "db.execute('create table trace(id integer)')\n"
                "db.commit(); db.close()\n"
                "(output_dir / 'trace.att').write_text('trace')\n"
                "(output_dir / 'gfx942_code_object_id_1.out').write_text('code')\n"
                "(output_dir / 'stats_ui_output_agent_0_dispatch_4.csv').write_text('name,value\\ngemm,1\\n')\n"
                "ui_dir = output_dir / 'ui_output_agent_0_dispatch_4'\n"
                "ui_dir.mkdir(parents=True)\n"
                "for name in ('code.json', 'filenames.json', 'occupancy.json', 'wstates0.json', 'se0_sm0_sl0_wv0.json'):\n"
                "    (ui_dir / name).write_text('{}')\n"
            )
            profiler.chmod(0o755)
            decoder = root / "librocprof-trace-decoder.so"
            decoder.write_text("decoder")

            profile = collect_amd_att(
                ("/usr/bin/python3", "-c", "pass"),
                root / "artifacts",
                kernel_filter="gemm",
                environment={
                    "TLX_ROCPROFV3": str(profiler),
                    "ROCPROF_ATT_LIBRARY_PATH": str(root),
                },
            )

            self.assertTrue(profile["valid"])
            self.assertEqual(profile["kernel_filter"], "gemm")
            self.assertEqual(profile["iteration_range"], "[4]")
            self.assertEqual(len(profile["artifacts"]["ui_directories"]), 1)

    def test_compares_numeric_stats_from_validated_att_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tlx_stats = root / "tlx.csv"
            aten_stats = root / "aten.csv"
            tlx_stats.write_text("metric,cycles,waves\ngemm,120,8\n")
            aten_stats.write_text("metric,cycles,waves\naten,100,4\n")
            comparison = compare_amd_att_profiles(
                {
                    "valid": True,
                    "validation": {"stats_files": [str(tlx_stats)]},
                },
                {
                    "valid": True,
                    "validation": {"stats_files": [str(aten_stats)]},
                },
            )
        self.assertTrue(comparison["valid"])
        self.assertEqual(comparison["tlx_over_aten"]["cycles"], 1.2)
        self.assertEqual(comparison["tlx_over_aten"]["waves"], 2.0)

    def test_summarizes_actionable_tlx_versus_aten_deltas(self) -> None:
        from ..decision_maker.tuning_harnesses.mm import _compare_with_aten

        comparison = _compare_with_aten(
            {
                "summary": {"duration_us": 120.0},
                "counters": {"MfmaUtil": 40.0, "MemUnitStalled": 8.0},
                "kernels": [{"resources": {"vgpr_count": 96.0}}],
            },
            {
                "summary": {"duration_us": 100.0},
                "counters": {"MfmaUtil": 80.0, "MemUnitStalled": 2.0},
                "kernels": [{"resources": {"vgpr_count": 48.0}}],
            },
            {"valid": True, "tlx_over_aten": {"Latency": 1.5}},
        )
        self.assertTrue(comparison["valid"])
        self.assertEqual(comparison["tlx_over_aten_duration"], 1.2)
        self.assertEqual(comparison["tlx_over_aten_counters"]["MfmaUtil"], 0.5)
        self.assertEqual(comparison["tlx_over_aten_resources"]["vgpr_count"], 2.0)

    def test_compact_profile_summary_omits_raw_blobs(self) -> None:
        compact = compact_profile_summary(
            {
                "level": "summary",
                "summary": {"duration_us": 1.0},
                "raw": "x" * 5000,
                "ncu": {
                    "raw_metrics": {"huge": "blob"},
                    "summary": {"duration_us": 1.0},
                },
                "rocprofv3": {
                    "summary": {"duration_us": 2.0},
                    "raw_metrics": {"huge": "blob"},
                },
                "amd_att": {
                    "valid": True,
                    "artifacts": {"ui_directories": ["/tmp/gemm_0_ui"]},
                },
                "amdgcn_isa": {
                    "resources": {"vgpr_spills": 37},
                    "findings": ["37 VGPR + 41 SGPR spills"],
                },
                "native_profiler": {
                    "raw": "x" * 5000,
                    "summary": {"duration_us": 1.0},
                },
                "diagnostic_proton_intra_kernel": {
                    "summary": {"active_warps": 8},
                    "trace_events": [{"raw": "event"}],
                    "artifacts": {"trace": "/tmp/proton.trace"},
                },
                "artifacts": {"csv": "/tmp/profile.csv"},
            }
        )
        self.assertEqual(compact["level"], "summary")
        self.assertNotIn("raw", compact)
        self.assertNotIn("raw_metrics", compact["ncu"])
        self.assertEqual(compact["rocprofv3"]["summary"]["duration_us"], 2.0)
        self.assertNotIn("raw_metrics", compact["rocprofv3"])
        self.assertTrue(compact["amd_att"]["valid"])
        self.assertEqual(compact["amdgcn_isa"]["resources"]["vgpr_spills"], 37)
        self.assertEqual(compact["amdgcn_isa"]["findings"], ["37 VGPR + 41 SGPR spills"])
        self.assertIn("native_profiler", compact)
        self.assertNotIn("raw", compact["native_profiler"])
        diagnostic = compact["diagnostic_proton_intra_kernel"]
        self.assertNotIn("trace_events", diagnostic)
        self.assertEqual(diagnostic["artifacts"]["trace"], "/tmp/proton.trace")
        self.assertEqual(compact["artifacts"]["csv"], "/tmp/profile.csv")


_AMDGCN_FIXTURE = Path(__file__).with_name("fixtures") / "amdgcn" / "persistent_mxfp8_excerpt.amdgcn"


class AmdgcnIsaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.analysis = analyze_amdgcn(
            _AMDGCN_FIXTURE.read_text(), kernel_name="_mxfp8_grouped_gemm_kernel"
        )

    def test_extracts_resources(self) -> None:
        resources = self.analysis["resources"]
        self.assertEqual(resources["total_vgprs"], 256)
        self.assertEqual(resources["vgpr_spills"], 37)
        self.assertEqual(resources["sgpr_spills"], 41)
        self.assertEqual(resources["scratch_bytes"], 152)
        self.assertEqual(resources["occupancy_waves_per_simd"], 2)

    def test_selects_innermost_mfma_loop_as_hot_loop(self) -> None:
        hot = self.analysis["hot_loop"]
        self.assertEqual(hot["label"], ".LBB0_16")
        self.assertEqual(hot["loop_depth"], 3)
        self.assertEqual(hot["mfma"], 8)
        self.assertEqual(hot["scaled_mfma"], 8)
        self.assertEqual(hot["ds_read_by_width"], {"b128": 4, "u8": 8})
        self.assertEqual(hot["byte_assembly_valu"], 3)
        self.assertEqual(hot["lds_dma_loads"], 2)
        self.assertEqual(hot["global_stores"], 0)

    def test_reports_reloads_that_drain_outstanding_stores(self) -> None:
        boundary = self.analysis["boundary"]
        self.assertGreater(boundary["global_stores"], 0)
        self.assertGreater(boundary["reload_drains_after_stores"], 0)
        reloads = self.analysis["source_locations"]["boundary_scratch_reloads"]
        self.assertTrue(reloads[0].startswith("gfx950.py:353 "))

    def test_attributes_byte_reads_to_source_lines(self) -> None:
        byte_reads = self.analysis["source_locations"]["hot_loop_byte_reads"]
        self.assertTrue(all(entry.startswith("gfx950.py:") for entry in byte_reads))
        self.assertFalse(any(":0 " in entry for entry in byte_reads))

    def test_findings_name_spills_reload_drains_and_byte_gathers(self) -> None:
        findings = "\n".join(self.analysis["findings"])
        self.assertIn("37 VGPR + 41 SGPR spills at the 256-VGPR ceiling", findings)
        self.assertIn("followed by s_waitcnt vmcnt(0) after", findings)
        self.assertIn("8 sub-dword ds_reads + 3 byte-assembly VALU per 8 MFMA", findings)
        self.assertLessEqual(len(self.analysis["findings"]), 8)

    def test_tolerates_dump_without_metadata(self) -> None:
        analysis = analyze_amdgcn(
            "k:\n.LBB0_1: ; =>This Inner Loop Header: Depth=1\n"
            "\tv_mfma_f32_16x16x32_bf16 v[0:3], v[4:7], v[8:11], v[0:3]\n"
            "\ts_cbranch_scc1 .LBB0_1\n.Lfunc_end0:\n",
            kernel_name="k",
        )
        self.assertEqual(analysis["resources"], {})
        self.assertEqual(analysis["hot_loop"]["mfma"], 1)
        self.assertEqual(analysis["findings"], [])

    def test_collects_dump_for_dominant_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "workload.py"
            script.write_text(
                "import os, pathlib, shutil, sys\n"
                "dump = pathlib.Path(os.environ['TRITON_DUMP_DIR']) / 'HASH'\n"
                "dump.mkdir(parents=True)\n"
                "assert os.environ['TRITON_ALWAYS_COMPILE'] == '1'\n"
                "shutil.copyfile(sys.argv[1], dump / '_mxfp8_grouped_gemm_kernel.amdgcn')\n"
                "(dump / 'other.amdgcn').write_text('')\n"
            )
            result = collect_amdgcn_isa(
                [sys.executable, str(script), str(_AMDGCN_FIXTURE)],
                Path(directory) / "artifacts",
                kernel_filter="_mxfp8_grouped_gemm_kernel.kd",
                timeout_seconds=60.0,
            )
            self.assertNotIn("error", result)
            self.assertEqual(result["tool"], "amdgcn_isa")
            self.assertEqual(result["kernel"], "_mxfp8_grouped_gemm_kernel")
            self.assertEqual(result["resources"]["vgpr_spills"], 37)
            self.assertTrue(Path(result["artifacts"]["amdgcn"]).is_file())

            missing = collect_amdgcn_isa(
                [sys.executable, str(script), str(_AMDGCN_FIXTURE)],
                Path(directory) / "artifacts-missing",
                kernel_filter="absent_kernel",
                timeout_seconds=60.0,
            )
            self.assertIn("no AMDGCN dump", missing["error"])
            self.assertIn("_mxfp8_grouped_gemm_kernel", missing["available_kernels"])

    def test_reports_failed_workload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = collect_amdgcn_isa(
                [sys.executable, "-c", "raise SystemExit(3)"],
                Path(directory),
                kernel_filter="k",
                timeout_seconds=60.0,
            )
        self.assertIn("failed with code 3", result["error"])


if __name__ == "__main__":
    unittest.main()
