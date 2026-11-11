"""Unit tests for runner_health (stdlib-only; run with pytest)."""
import datetime as dt

import runner_health as rh

NOW = dt.datetime(2026, 11, 2, 18, tzinfo=dt.timezone.utc)
POOL = "nvidia-dgx-b200"


def job(status, minutes_ago, wf="gpu-tests.yml"):
    at = (NOW - dt.timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")
    return {"status": status, "created_at": at, "labels": [POOL], "wf": wf, "html_url": f"job-{minutes_ago}"}


def test_pool_is_down_only_when_its_jobs_wait_and_none_runs():
    waiting = job("queued", 60)
    down, _ = rh.pool_states([waiting, job("queued", 30)], NOW)
    assert down == {POOL: waiting}
    # A busy pool queues jobs too, but it is running one.
    down, running = rh.pool_states([waiting, job("in_progress", 20, wf="tritonbench.yml")], NOW)
    assert down == {} and running == {POOL}
