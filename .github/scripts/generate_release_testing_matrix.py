import argparse
import json
import os
import sys

# Suites that can be released-tested, and the runners each one supports.
SUITE_CONFIG = {
    "tritonbench": {
        "runners": ["h100", "mi350", "b200"],
    },
}

RUNNER_FULL_NAMES = {
    "h100": "linux-gcp-h100",
    "mi350": "linux-fb-triton-mi350-1",
    "b200": "nvidia-dgx-b200",
}

SUPPORTED_RUNNERS = {"h100", "mi350", "b200", "all"}
SUPPORTED_TEST_TYPES = {"periodic", "single", "abtest"}


def parse_csv(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_runners(runners: list[str]) -> list[str]:
    if not runners or "all" in runners:
        return []
    return runners


def validate_requested_values(
    suites: list[str],
    runners: list[str],
    test_type: str,
) -> None:
    unknown_suites = [name for name in suites if name not in SUITE_CONFIG]
    if unknown_suites:
        raise ValueError(f"Unsupported suites: {', '.join(sorted(unknown_suites))}")

    unknown_runners = [runner for runner in runners if runner not in SUPPORTED_RUNNERS]
    if unknown_runners:
        raise ValueError(f"Unsupported runners: {', '.join(sorted(unknown_runners))}")

    if test_type not in SUPPORTED_TEST_TYPES:
        raise ValueError(f"Unsupported test type: {test_type}")


def select_suites(requested_suites: list[str]) -> list[str]:
    if requested_suites:
        return requested_suites
    return ["tritonbench"]


def filter_dimensions(
    suite: str,
    requested_runners: list[str],
) -> list[dict[str, str]]:
    config = SUITE_CONFIG[suite]
    runners = list(config["runners"])

    if requested_runners:
        requested_runner_set = set(requested_runners)
        runners = [runner for runner in runners if runner in requested_runner_set]

    matrix_entries = []
    for runner in runners:
        matrix_entries.append({
            "suite": suite,
            "runner": runner,
            "runner_full_name": RUNNER_FULL_NAMES[runner],
        })
    return matrix_entries


def to_matrix(entries: list[dict[str, str]]) -> str:
    return json.dumps({"include": entries}, separators=(",", ":"))


def write_output(name: str, value: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
    else:
        print(f"{name}={value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="tritonbench")
    parser.add_argument("--runners", default="")
    parser.add_argument("--test-type", default="periodic")
    parser.add_argument("--event-name", default="")
    args = parser.parse_args()

    requested_suites = parse_csv(args.suite)
    requested_runners = normalize_runners(parse_csv(args.runners))

    validate_requested_values(requested_suites, requested_runners, args.test_type)

    suites = select_suites(requested_suites)

    full_matrix_entries: list[dict[str, str]] = []
    for suite in suites:
        full_matrix_entries.extend(filter_dimensions(
            suite,
            requested_runners,
        ))

    write_output("test_matrix", to_matrix(full_matrix_entries))
    write_output("has_test_suites", str(bool(full_matrix_entries)).lower())

    if not full_matrix_entries:
        sys.stderr.write("No benchmark matrix entries were generated.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
