from tlx_ops_bench_pick import pick


def run(sha, **jobs):
    return {
        "head_sha": sha, "jobs": {"build / build": "success", **{f"{p} / {p}-tlx-test": c
                                                                 for p, c in jobs.items()}}
    }


def test_red_or_untested_platform_never_blocks_the_bench():
    runs = [run("untested", h100="success"), run("red", h100="failure", mi350="success")]
    chosen, ci = pick(runs, lambda r: r["jobs"], ["h100", "mi350"])
    assert chosen["head_sha"] == "red" and ci == {"h100": "failure", "mi350": "success"}
    chosen, ci = pick(runs[:1], lambda r: r["jobs"], ["h100", "mi350"])
    assert chosen["head_sha"] == "untested" and ci == {"h100": "success", "mi350": "not run"}
