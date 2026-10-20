# GPU CI

`workflows/gpu-tests.yml` selects platforms, calls `build.yml` once, then passes
the same wheel URL and SHA256 to `h100.yml`, `b200.yml`, and `mi350.yml`. Build and
test checkouts use the same exact commit, including the merge commit for PRs.
The GPU workflows are reusable test workflows; trigger the parent to run them.
Fork PRs pass the wheel through GitHub artifacts and skip AWS and the S3 cache;
their downloads are checksum-verified too.

Push and PR platform selection uses the ordered rules in `gpu-test-paths.json`,
preserving the former individual workflow filters. PRs use a three-dot diff;
pushes compare the before/after SHAs. An unavailable diff runs all platforms.
Schedules run every six hours at minute zero, and manual runs also select all
platforms. Documentation-only changes run selection and the aggregate check,
without building or allocating GPU runners.

## Checks and nightly reporting

Use **GPU tests** as the aggregate required check: a failed build or selected
suite fails it, while intentionally unselected suites do not. Reusable workflows
prefix individual check names, for example `h100 / h100-tlx-test`; branch rules
that require the old bare job names need to use the new names or the aggregate.
The nightly publisher accepts both old and new TLX check names during migration.

Nightly issue titles retain the previous platform workflow names. Last-green
lookup checks the relevant GPU job independently of sibling failures and includes
the old workflow's history. Compiler nightly reporting is unchanged.

## Wheel reuse

`scripts/wheel_cache.py` stores a manifest under
`s3://gha-artifacts/tritonci-artifacts/wheel-cache/v1/`, keyed by repository,
resolved SHA, Python ABI, runner platform, and build recipe. A ready manifest
records the uploaded wheel's key, size, checksum, and expiry. Cache hits verify
that the object exists with the expected size; every GPU job verifies the
downloaded checksum before installing. Missing or expired artifacts rebuild.
Wheels retain the existing 14-day retention; reuse stops four hours before
expiry to allow downstream jobs time to download them.

Benchmark callers retain the repository, build ID, and resolved-commit interface.
Callers setting `skip_cache_save` bypass the wheel cache as well as GitHub cache
writes, and upload their wheel directly. Build IDs keep side A/B uploads distinct.

Conditional S3 PUTs (`If-None-Match` / `If-Match`) provide one producer per cache
key. Waiters poll every 30 seconds for up to 90 minutes. The producer publishes
the ready manifest only after wheel validation and upload. An `always()` cleanup
releases failed builds; a lost runner's lease expires after four hours. Lease
duration must remain longer than the build job's entire timeout. An old producer
cannot publish over or release a replacement producer's lease.

The existing `gha_workflow_triton_artifacts` role needs `s3:GetObject` and
`s3:PutObject` on the cache prefix as well as the existing artifact permissions.
It also needs `s3:ListBucket` for missing keys to return 404 instead of 403. No
delete permission is needed for the cache. Authorization or service errors fail
the build rather than silently bypassing the producer lock.

## CI health dashboard

`/ci/` on the Pages site shows four tabs, one owner each:

- **Main** (on-call, commit author): has a push to `main` broken something new?
  A failing test is *new* if it passed on the previous push that ran its test
  step on the same GPU, and *known* if it failed there too. A known failure
  keeps the push it first failed at. A step that was skipped or hit a setup
  failure has no result and does not count.
- **Nightly & perf** (release and perf owners): the `nightly-fbtriton.yml`
  release and the perf events from `/bench/`.
- **Pull requests** (PR author): the latest run of each PR, tagged *fails on
  main* or *setup: CI* when the failure is not the author's.
- **Operations** (CI maintainers): GPU hours, job durations, setup failures,
  and what the `select` job ran per trigger, with runs where it failed, fell
  back to every suite, or skipped a GPU on a scheduled or manual run.

`ci-health.yml` runs `scripts/ci_health.py collect` every three hours. It stores
`gpu-tests.yml` and `nightly-fbtriton.yml` runs, with jobs and steps, on the
`ci-health` release. For each failed GPU job it reads the failing tests from the
job log, and from each `select` job log it reads the selection
(`scripts/ci_health_logs.py`). `pages.yml` then runs
`ci_health.py render`, which computes `data.json` with
`scripts/ci_health_rules.py`. The page is `scripts/ci_health_dashboard.html`.

The dashboard depends on these properties of the GPU workflows:

- GPU job check names keep the `h100 / `, `b200 / ` or `mi350 / ` prefix. The
  prefix is how a job maps to a GPU.
- Every test step's name matches a pattern in `GROUPS` in `ci_health_rules.py`.
  Any other step counts as setup.
  `test_every_test_step_in_the_gpu_workflows_maps_to_a_group` fails when a
  `Run ...` step has no group.
- pytest prints its short test summary (`FAILED <nodeid> - <message>`). Flags
  that hide it, such as `-rN`, make every failure look like a crash of the whole
  step. Parametrized variants count as one test.
- Setup steps (install, download, checkout) match `INFRA_STEP`, so their
  failures are counted as CI's and not as test failures.
- The shared jobs keep the names `select` and `build` (its checks show as
  `build / build`). Main shows their failure as a build failure.
- `gpu_test_selection.py` keeps printing `GPU selection for <sha>: {...}`, and
  `Cannot determine changed paths` when it falls back to every suite.

To re-collect after a fix to the collector, run **CI Health** manually with
`since` set to an ISO time. GitHub keeps job logs for 90 days, so failures older
than that cannot be recovered.

## Local validation

```bash
python3 -m pytest .github/scripts/test_gpu_ci.py .github/scripts/test_nightly_select.py -s --tb=short
(cd .github/scripts && python3 -m pytest -q test_ci_health.py test_ci_health_rules.py test_ci_health_logs.py)
node --test .github/scripts/test_last_green_nightly.js
```

The Python tests need pytest, PyYAML, and boto3. They exercise conditional-write
races, failure recovery, expiry, path selection, and workflow dependencies
without contacting AWS or using a GPU. Hosted CI must validate the actual role's
S3 access and installation of the shared wheel on each GPU runner.
