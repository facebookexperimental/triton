"""Unit tests for runner_health (stdlib-only; run with pytest)."""
import datetime as dt

import runner_health as rh

NOW = dt.datetime(2026, 11, 2, 18, tzinfo=dt.timezone.utc)
POOL = "nvidia-dgx-b200"


def job(status, minutes_ago, wf="gpu-tests.yml", run_id=1, labels=(POOL, )):
    at = (NOW - dt.timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")
    return {
        "status": status, "created_at": at, "labels": list(labels), "wf": wf, "html_url": f"job-{minutes_ago}",
        "run_id": run_id
    }


def test_pool_is_down_only_when_its_jobs_wait_and_none_runs():
    waiting = job("queued", 60)
    down, _ = rh.pool_states([waiting, job("queued", 30)], NOW)
    assert down == {POOL: waiting}
    # A busy pool queues jobs too, but it is running one.
    down, running = rh.pool_states([waiting, job("in_progress", 20, wf="tritonbench.yml")], NOW)
    assert down == {} and running == {POOL}


def test_run_is_cancelled_only_when_it_waits_on_nothing_but_a_down_pool_past_the_timeout():
    late = rh.QUEUE_TIMEOUT_MIN + 1
    jobs = [
        job("queued", late, run_id=1),
        job("completed", late, run_id=1, labels=["linux-gcp-h100"]),
        # Another GPU of the run is still testing: let it finish first.
        job("queued", late, run_id=2),
        job("in_progress", 5, run_id=2, labels=["linux-gcp-h100"]),
        # Down, but not for long enough.
        job("queued", rh.QUEUE_TIMEOUT_MIN - 1, run_id=3),
        # Not a CI test workflow.
        job("queued", late, run_id=4, wf=rh.rules.BENCH_WF),
    ]
    down, _ = rh.pool_states(jobs, NOW)
    assert list(rh.stuck_runs(jobs, down, NOW)) == [1]
    # A busy pool is running a job, so nothing waiting on it is cancelled.
    busy = jobs + [job("in_progress", 5, run_id=5, wf="tritonbench.yml")]
    down, _ = rh.pool_states(busy, NOW)
    assert rh.stuck_runs(busy, down, NOW) == {}
