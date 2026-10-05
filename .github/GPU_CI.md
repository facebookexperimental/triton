# GPU CI

`workflows/gpu-tests.yml` selects platforms, calls `build.yml` once, then passes
the same wheel URL and SHA256 to `h100.yml`, `b200.yml`, and `mi350.yml`. Build and
test checkouts use the same exact commit, including the merge commit for PRs.
The GPU workflows are reusable test workflows; trigger the parent to run them.

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

## Local validation

```bash
python3 -m pytest .github/scripts/test_gpu_ci.py .github/scripts/test_nightly_select.py -s --tb=short
node --test .github/scripts/test_last_green_nightly.js
```

The Python tests need pytest, PyYAML, and boto3. They exercise conditional-write
races, failure recovery, expiry, path selection, and workflow dependencies
without contacting AWS or using a GPU. Hosted CI must validate the actual role's
S3 access and installation of the shared wheel on each GPU runner.
