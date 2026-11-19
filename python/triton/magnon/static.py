"""compile_iq hooks for launchers that bypass CompiledKernel.run (e.g. Inductor's static CUDA launcher).

Collection and the plain-vs-ACF A/B normally run on a kernel's first launch through Triton's own
launcher (collector.wrap_launcher, CompiledKernel._compile_iq_resolve). A launcher that loads and
launches the cubin itself never reaches them, so it takes a LaunchHooks from make_launch_hooks() when
the kernel is compiled and calls first_launch() on the first launch for which ready() holds.

LaunchHooks is plain picklable data, so it travels with the launcher from a compile worker or through
a compile cache. The ACF store is read at first_launch, in the launching process, so -- as on Triton's
cache-hit path -- a cached kernel still picks up a newly tuned ACF. Everything here is fail-open.
"""

import dataclasses
import os
import tempfile

from triton.runtime.jit import _compile_iq_state

from .flags import apply_enabled, collect_enabled
from .store import dlog


@dataclasses.dataclass
class LaunchHooks:
    name: str
    fn_name: str | None
    ptx: str
    metadata: dict  # the kernel's metadata, as in its compile-cache .json (target as a dict)
    arg_names: list  # every kernel arg, in order
    constexprs: list  # the source's constexpr arg paths
    asm: dict  # optional "launch_metadata" / "launcher_src" entries collected with the task
    collect: bool
    apply: bool

    def ready(self) -> bool:
        """False while launches are benchmark runs (collector.benchmarking(), which Triton's autotuner
        and Inductor's benchmarker enter) or the stream is capturing a CUDA graph."""
        if getattr(_compile_iq_state, "suppressed", False):
            return False
        try:
            import torch

            return not torch.cuda.is_current_stream_capturing()
        except Exception:
            return True

    def first_launch(self, grid, call_args, *, plain, launch, load_cubin):
        """Collect the task and run the ACF A/B for the kernel's first real launch, before the caller
        launches. Returns the ACF function handle if it should replace `plain`, else None.

        call_args: the launch's runtime args by name, in launch order (constexprs may be omitted), or
            None when the caller cannot name them; collection is then skipped.
        plain: the caller's live function handle.
        launch(function, args): launches `function` on the caller's grid and stream with positional
            args in call_args order (copies of the caller's, during the A/B).
        load_cubin(path): loads a cubin file and returns its function handle; the caller owns the
            module and unloads it if None is returned. None skips the A/B.
        """
        metadata = None
        if self.collect:
            if call_args is None:
                dlog("collector", f"skip {self.name} (launcher cannot name its args)")
            else:
                try:
                    from .collector import capture_compiled

                    metadata = _kernel_metadata(self.metadata)
                    capture_compiled(ptx=self.ptx, metadata=metadata, asm=self.asm, kernel_name=self.name,
                                     fn_name=self.fn_name, bound_args={n: call_args.get(n)
                                                                       for n in self.arg_names},
                                     constexprs=self.constexprs, grid=grid)
                except Exception as e:  # collection must never break the user's run
                    dlog("collector", f"capture failed: {type(e).__name__}: {e}")
        if not self.apply or load_cubin is None or call_args is None:
            return None
        try:
            from triton.compiler.compiler import make_backend

            from .consume import ab_keeps_acf

            metadata = metadata or _kernel_metadata(self.metadata)
            backend = make_backend(metadata.target)
            if not hasattr(backend, "compile_iq_acf_cubin"):
                return None
            cubin = backend.compile_iq_acf_cubin(self.ptx, metadata)
            if cubin is None:
                return None
            with tempfile.TemporaryDirectory(prefix="compile_iq_") as d:
                path = os.path.join(d, f"{self.name}.cubin")
                with open(path, "wb") as f:
                    f.write(cubin)
                acf = load_cubin(path)
            return acf if ab_keeps_acf(self.name, launch, plain, acf, list(call_args.values())) else None
        except Exception as e:  # never break the user's launch
            if os.environ.get("TRITON_COMPILE_IQ_DEBUG"):
                print(f"[compile_iq.freewin] {self.name}: error (fail-open, kept plain): "
                      f"{type(e).__name__}: {e}", flush=True)
            return None


def make_launch_hooks(kernel):
    """LaunchHooks for a CompiledKernel, or None when compile_iq is off or the kernel has no PTX."""
    collect, apply = collect_enabled(), apply_enabled()
    if not (collect or apply):
        return None
    try:
        asm = kernel.asm
        if "ptx" not in asm:
            return None
        metadata = kernel.metadata._asdict()
        target = metadata["target"]
        metadata["target"] = {"backend": target.backend, "arch": target.arch, "warp_size": target.warp_size}
        src = kernel.src
        fn = src.fn
        return LaunchHooks(
            name=kernel.name,
            fn_name=getattr(fn, "__name__", None),
            ptx=asm["ptx"],
            metadata=metadata,
            arg_names=list(fn.arg_names),
            constexprs=list(getattr(src, "constants", None) or {}),
            asm={k: asm[k]
                 for k in ("launch_metadata", "launcher_src")
                 if k in asm},
            collect=collect,
            apply=apply,
        )
    except Exception as e:
        dlog("collector", f"no launch hooks for {getattr(kernel, 'name', '?')}: {type(e).__name__}: {e}")
        return None


def _kernel_metadata(metadata: dict):
    """Rebuild a CompiledKernel-style metadata namedtuple (see CompiledKernel.__init__)."""
    from collections import namedtuple

    from triton.backends.compiler import GPUTarget

    md = dict(metadata)
    target = md["target"]
    md["target"] = GPUTarget(target["backend"], target["arch"], target["warp_size"])
    return namedtuple("KernelMetadata", sorted(md))(**md)
