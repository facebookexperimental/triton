# Triton

Upstream Triton compiler. See `examples/plugins/` for the extension plugin system.

## Development

- Rebuild after C++ changes: `pip install -e . --no-build-isolation`
- Build with shared libs for plugin development: `export LLVM_BUILD_SHARED_LIBS=1`
