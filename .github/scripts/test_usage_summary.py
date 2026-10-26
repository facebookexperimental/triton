"""Unit tests for usage_summary (stdlib-only; run with pytest)."""

import usage_summary as us

HEADER = "time,cpu_usage_usec,memory_bytes,gpu_util_pct,gpu_memory_mib,gpu_memory_total_mib,gpu_power_w,gpu_sm_clock_mhz\n"


def write(tmp_path, rows):
    path = tmp_path / "usage.csv"
    path.write_text(HEADER + "".join(rows))
    return str(path)


def test_cpu_cores_from_usage_deltas(tmp_path):
    # 2 s of wall time, 4 s of CPU time per second -> 4 cores busy.
    path = write(tmp_path, [
        "100.0,0,1073741824,50,1024,183359,300,1965\n",
        "101.0,4000000,2147483648,100,2048,183359,600,1965\n",
        "102.0,8000000,1073741824,0,1024,183359,200,1965\n",
    ])
    samples = us.series(us.load(path))
    assert [s["cpu_cores"] for s in samples] == [None, 4.0, 4.0]
    assert samples[1]["memory"] == 2.0
    assert samples[1]["gpu_memory"] == 2.0


def test_report_stats_and_busy_share(tmp_path):
    path = write(
        tmp_path,
        [f"{100 + i}.0,{i * 1000000},1073741824,{u},1024,183359,300,1965\n" for i, u in enumerate([0, 95, 100, 50])])
    text = us.report(us.series(us.load(path)), cpus=224, gpus=1, memory_max="max", memory_peak=2 * us.GIB)
    assert "4 samples over 3 s" in text
    assert "| GPU utilization (%) | 61.2 | 50.0 | 100.0 | 100.0 |" in text
    assert "| CPU busy (cores) | 1.0 | 1.0 | 1.0 | 1.0 |" in text
    assert "at least 90% in 50% of samples and 0% in 25%" in text
    assert "memory peak 2.0 GiB" in text


def test_missing_gpu_columns(tmp_path):
    # MI350 has no nvidia-smi, so the GPU columns are empty.
    path = write(tmp_path, ["100.0,0,1073741824,,,,,\n", "101.0,2000000,1073741824,,,,,\n"])
    text = us.report(us.series(us.load(path)), cpus=8, gpus=0, memory_max="max", memory_peak=0)
    assert "| GPU utilization (%) | n/a | n/a | n/a | n/a |" in text
    assert "| CPU busy (cores) | 2.0 | 2.0 | 2.0 | 2.0 |" in text
    assert "at least 90%" not in text


def test_per_minute_timeline(tmp_path):
    rows = [f"{t}.0,{t * 1000000},1073741824,{10 if t < 60 else 90},1024,183359,300,1965\n" for t in range(0, 120, 10)]
    text = us.report(us.series(us.load(write(tmp_path, rows))), cpus=8, gpus=1, memory_max="max", memory_peak=0)
    assert "| 0 | 10.0 |" in text
    assert "| 1 | 90.0 |" in text


def test_too_few_samples(tmp_path):
    text = us.report(us.series(us.load(write(tmp_path, ["100.0,0,1,0,0,0,0,0\n"]))), 8, 1, "max", 0)
    assert "Fewer than two samples" in text


def test_missing_file_does_not_fail(tmp_path, capsys):
    assert us.main([str(tmp_path / "nope.csv")]) == 0
    assert "No samples" in capsys.readouterr().out
