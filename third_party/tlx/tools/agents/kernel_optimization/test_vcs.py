from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from .vcs import ATTRIBUTION, AutoCommitError, commit_winner, prepare_auto_commit


def _run(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=True
    ).stdout


@unittest.skipUnless(shutil.which("git"), "git is required")
class GitAutoCommitTest(unittest.TestCase):
    def _repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        _run(["git", "init", "-q"], root)
        _run(["git", "config", "user.name", "TLX Test"], root)
        _run(["git", "config", "user.email", "tlx@example.com"], root)
        kernel = root / "kernels" / "kernel.py"
        kernel.parent.mkdir()
        kernel.write_text("A = 1\nKEEP = 1\nKEEP2 = 1\nB = 1\n")
        (root / "other.txt").write_text("base\n")
        _run(["git", "add", "."], root)
        _run(["git", "commit", "-qm", "base"], root)
        return temporary, root, kernel

    def test_commits_only_winner_delta_and_preserves_other_work(self) -> None:
        temporary, root, kernel = self._repo()
        self.addCleanup(temporary.cleanup)
        kernel.write_text("A = 2\nKEEP = 1\nKEEP2 = 1\nB = 1\n")
        baseline = kernel.read_text()
        (root / "other.txt").write_text("staged\n")
        _run(["git", "add", "other.txt"], root)
        (root / "dirty.txt").write_text("dirty\n")
        snapshot = prepare_auto_commit(kernel, baseline)

        validated: list[str] = []
        result = commit_winner(
            snapshot,
            "A = 2\nKEEP = 1\nKEEP2 = 1\nB = 2\n",
            "Tune kernel",
            body="Fold the scale into the exponent.\n\nTLX agent authored",
            validate_committed_source=validated.append,
        )

        self.assertTrue(result.success)
        self.assertEqual(
            validated,
            ["A = 1\nKEEP = 1\nKEEP2 = 1\nB = 2\n"],
        )
        self.assertEqual(
            _run(["git", "show", "HEAD:kernels/kernel.py"], root),
            "A = 1\nKEEP = 1\nKEEP2 = 1\nB = 2\n",
        )
        self.assertEqual(
            kernel.read_text(), "A = 2\nKEEP = 1\nKEEP2 = 1\nB = 2\n"
        )
        self.assertIn("other.txt", _run(["git", "diff", "--cached", "--name-only"], root))
        self.assertEqual(
            _run(["git", "show", "--format=", "--name-only", "HEAD"], root).strip(),
            "kernels/kernel.py",
        )
        message = _run(["git", "log", "-1", "--format=%B"], root)
        self.assertIn("Fold the scale into the exponent.", message)
        self.assertEqual(message.count(ATTRIBUTION), 1)
        self.assertTrue(message.rstrip().endswith(ATTRIBUTION))

    def test_rejects_dirty_target_without_merged_source_validation(self) -> None:
        temporary, root, kernel = self._repo()
        self.addCleanup(temporary.cleanup)
        kernel.write_text("A = 2\nKEEP = 1\nKEEP2 = 1\nB = 1\n")
        snapshot = prepare_auto_commit(kernel, kernel.read_text())
        before = _run(["git", "rev-parse", "HEAD"], root)

        with self.assertRaisesRegex(AutoCommitError, "requires validation"):
            commit_winner(
                snapshot,
                "A = 2\nKEEP = 1\nKEEP2 = 1\nB = 2\n",
                "Tune kernel",
            )

        self.assertEqual(before, _run(["git", "rev-parse", "HEAD"], root))
        self.assertEqual(
            kernel.read_text(), "A = 2\nKEEP = 1\nKEEP2 = 1\nB = 1\n"
        )

    def test_rejects_dirty_target_when_merged_source_validation_fails(self) -> None:
        temporary, root, kernel = self._repo()
        self.addCleanup(temporary.cleanup)
        kernel.write_text("A = 2\nKEEP = 1\nKEEP2 = 1\nB = 1\n")
        snapshot = prepare_auto_commit(kernel, kernel.read_text())
        before = _run(["git", "rev-parse", "HEAD"], root)

        def reject(source: str) -> None:
            self.assertEqual(source, "A = 1\nKEEP = 1\nKEEP2 = 1\nB = 2\n")
            raise AutoCommitError("merged source failed correctness")

        with self.assertRaisesRegex(AutoCommitError, "failed correctness"):
            commit_winner(
                snapshot,
                "A = 2\nKEEP = 1\nKEEP2 = 1\nB = 2\n",
                "Tune kernel",
                validate_committed_source=reject,
            )

        self.assertEqual(before, _run(["git", "rev-parse", "HEAD"], root))
        self.assertEqual(
            kernel.read_text(), "A = 2\nKEEP = 1\nKEEP2 = 1\nB = 1\n"
        )

    def test_rejects_overlapping_dirty_target(self) -> None:
        temporary, root, kernel = self._repo()
        self.addCleanup(temporary.cleanup)
        kernel.write_text("A = 2\nKEEP = 1\nKEEP2 = 1\nB = 1\n")
        snapshot = prepare_auto_commit(kernel, kernel.read_text())
        before = _run(["git", "rev-parse", "HEAD"], root)
        with self.assertRaisesRegex(AutoCommitError, "overlaps"):
            commit_winner(
                snapshot, "A = 3\nKEEP = 1\nKEEP2 = 1\nB = 1\n", "Tune kernel"
            )
        self.assertEqual(before, _run(["git", "rev-parse", "HEAD"], root))
        self.assertEqual(
            kernel.read_text(), "A = 2\nKEEP = 1\nKEEP2 = 1\nB = 1\n"
        )

    def test_rejects_target_change_during_optimization(self) -> None:
        temporary, _root, kernel = self._repo()
        self.addCleanup(temporary.cleanup)
        snapshot = prepare_auto_commit(kernel, kernel.read_text())
        kernel.write_text("A = 9\nB = 1\n")
        with self.assertRaisesRegex(AutoCommitError, "changed during optimization"):
            commit_winner(snapshot, "A = 1\nB = 2\n", "Tune kernel")


@unittest.skipUnless(shutil.which("hg"), "hg is required")
class HgAutoCommitTest(unittest.TestCase):
    def test_commits_only_target_and_preserves_dirty_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _run(["hg", "init"], root)
            kernel = root / "kernel.py"
            kernel.write_text("A = 1\nKEEP = 1\nKEEP2 = 1\nB = 1\n")
            other = root / "other.txt"
            other.write_text("base\n")
            _run(["hg", "add", "kernel.py", "other.txt"], root)
            _run(["hg", "commit", "-u", "TLX Test", "-m", "base"], root)
            kernel.write_text("A = 2\nKEEP = 1\nKEEP2 = 1\nB = 1\n")
            other.write_text("dirty\n")
            snapshot = prepare_auto_commit(kernel, kernel.read_text())

            validated: list[str] = []
            result = commit_winner(
                snapshot,
                "A = 2\nKEEP = 1\nKEEP2 = 1\nB = 2\n",
                "Tune kernel",
                validate_committed_source=validated.append,
            )

            self.assertTrue(result.success)
            self.assertEqual(
                validated,
                ["A = 1\nKEEP = 1\nKEEP2 = 1\nB = 2\n"],
            )
            self.assertEqual(
                _run(["hg", "cat", "-r", ".", "kernel.py"], root),
                "A = 1\nKEEP = 1\nKEEP2 = 1\nB = 2\n",
            )
            self.assertEqual(
                kernel.read_text(), "A = 2\nKEEP = 1\nKEEP2 = 1\nB = 2\n"
            )
            status = _run(["hg", "status"], root)
            self.assertIn("M kernel.py", status)
            self.assertIn("M other.txt", status)
            self.assertIn(ATTRIBUTION, _run(["hg", "log", "-r", ".", "--template", "{desc}"], root))


if __name__ == "__main__":
    unittest.main()
