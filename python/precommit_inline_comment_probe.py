"""Temporary fork-PR reporting probe. Do not land this test-only change.

After the reporter workflow lands on the GitHub default branch, open a fork PR
with this file. Expect pre-commit-strict to fail and the App to post an inline
F821 comment on the return statement below. Rerunning the same commit should
not duplicate that comment. Remove this file after verifying the report.
"""


def precommit_inline_comment_probe():
    return intentional_undefined_name_for_fork_inline_comment
