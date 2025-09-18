# FB-only:

this PR will be imported to fbcode diff. ensure all internal tests passed.

run `third_party/tlx/run_all.sh` to cover TLX unit tests and TLX kernel correctness verification:

```
Need to build triton in this script? {y|n}n
Run all LITs? {y|n}n
Run core Triton python unit tests? {y|n}n
Run all TLX unit tests? {y|n}y
Running TLX Unit Tests
...
Run TLX tutorial kernels (correctness|performance|no)? {c|p|n}c
...
```

# New contributor declaration (copied from Core Triton):
- [ ] I am not making a trivial change, such as fixing a typo in a comment.

- [ ] I have written a PR description following these
  [rules](https://cbea.ms/git-commit/#why-not-how).

- [ ] I have run `pre-commit run --from-ref origin/main --to-ref HEAD`.

- Select one of the following.
  - [ ] I have added tests.
    - `/test` for `lit` tests
    - `/unittest` for C++ tests
    - `/python/test` for end-to-end tests
  - [ ] This PR does not need a test because `FILL THIS IN`.

- Select one of the following.
  - [ ] I have not added any `lit` tests.
  - [ ] The `lit` tests I have added follow these [best practices](https://mlir.llvm.org/getting_started/TestingGuide/#filecheck-best-practices),
    including the "tests should be minimal" section. (Usually running Python code
    and using the instructions it generates is not minimal.)
