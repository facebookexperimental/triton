# DO NOT LAND: a deliberate failure to check the CI failure summary (#4357).


def test_ci_failure_summary_canary():
    assert 1 + 1 == 3, "deliberate failure: checks the run page summary and the internal CI signal card"
