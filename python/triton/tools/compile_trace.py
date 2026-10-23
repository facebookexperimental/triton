"""Perfetto/Chrome trace of Triton compilation, across processes.

Enabled by TRITON_COMPILE_TRACE=<file or directory>. Given a directory
(existing, or ending in "/"), each run writes a new
triton_compile_trace_<timestamp>_<pid>.json in it; given a file, that file is
appended to. Missing directories are created. Every process of a run writes to the same file, so the compiling
process, its pool workers and the ptxas they spawn render on one timeline.
Open the file in https://ui.perfetto.dev.

What is recorded:
  * one span per compile (cache key, cache lookup, ast, each lowering stage,
    cache store) and per kernel load, each naming the function that
    ran in its args,
  * MLIR passes, LLVM O3 passes and LLVM codegen passes inside each stage,
    from LLVM's time-trace profiler,
  * each subprocess run through run_subprocess (ptxas) on a "subprocess" track
    of the process that spawned it, with its pid, CPU time and peak RSS, plus
    an RSS counter sampled while it runs,
  * process RSS/anon at stage boundaries.

The file uses the JSON Array trace format, whose closing bracket is optional;
each process appends its events with one O_APPEND write per flush, so
processes never interleave within a batch. Timestamps are CLOCK_MONOTONIC,
which all processes on a host share. The first process of a run exports the
file it resolved as TRITON_COMPILE_TRACE_FILE, which the processes it starts
inherit, so they join its trace rather than starting their own. Appending
relies on local-filesystem O_APPEND semantics; independent runs should use a
directory rather than share a file.

Tracing never fails a compile: an error in the tracer itself disables tracing
with a warning. Setting add_stages_inspection_hook disables the JIT's native
fast launch path (see runtime/jit.py), so plain @triton.jit launches are
slower while tracing.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import warnings

_FILE_ENV = "TRITON_COMPILE_TRACE_FILE"

_path: str | None = None
_lock = threading.Lock()
_events: list[dict] = []
_named_pid: int | None = None
_load_starts: dict[str, float] = {}
# Times observed during the compile running on this thread: when the cache key
# is done and the stages are added (both via add_stages_inspection_hook), and
# the first stage start and last stage end.
_compile = threading.local()


# Native events shorter than this are dropped; large kernels otherwise produce
# thousands of per-function LLVM events.
def _env_number(name: str, default: float) -> float:
    # Read while triton is imported, so a bad value must not raise.
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        warnings.warn(f"TRITON_COMPILE_TRACE: ignoring invalid {name}, using {default}")
        return default


_GRANULARITY_US = int(_env_number("TRITON_COMPILE_TRACE_GRANULARITY_US", 50))
_SAMPLE_S = _env_number("TRITON_COMPILE_TRACE_SAMPLE_MS", 10) / 1000
_PAGE_MB = os.sysconf("SC_PAGE_SIZE") / 2**20
# Perfetto groups threads by (pid, tid); subprocesses get a synthetic thread
# per spawning process, as that process runs its subprocesses one at a time.
_SUBPROCESS_TID_BIT = 1 << 30
# Starts and ends of traced subprocesses in this process, to tell whether a
# RUSAGE_CHILDREN delta belongs to one subprocess alone.
_subprocess_events = 0


def _disable(error: Exception) -> None:
    global _path
    _path = None
    warnings.warn(f"TRITON_COMPILE_TRACE: tracing disabled after an internal error: {error!r}")


def _guarded(fn):
    """Runs tracer bookkeeping; a failure disables tracing instead of failing the compile."""

    @functools.wraps(fn)
    def run(*args, **kwargs):
        if _path is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            _disable(e)
            return None

    return run


def _now_us() -> float:
    return time.monotonic_ns() / 1000


def _emit(event: dict) -> None:
    event.setdefault("pid", os.getpid())
    event.setdefault("tid", threading.get_native_id())
    with _lock:
        _events.append(event)


def _span(name: str, start_us: float, end_us: float, cat: str, args: dict | None = None, **kw) -> None:
    _emit({"name": name, "cat": cat, "ph": "X", "ts": start_us, "dur": end_us - start_us, "args": args or {}, **kw})


def _counter(name: str, ts_us: float, values: dict) -> None:
    _emit({"name": name, "ph": "C", "ts": ts_us, "args": values})


def _memory_counter(ts_us: float) -> None:
    with open("/proc/self/statm") as f:
        _, resident, shared, *_ = f.read().split()
    rss = int(resident) * _PAGE_MB
    _counter("memory (MB)", ts_us, {"rss": round(rss, 1), "anon": round(rss - int(shared) * _PAGE_MB, 1)})


def _role() -> str:
    # Inductor marks its pool workers; its pool process runs compile_worker.
    utils = sys.modules.get("torch._inductor.compile_worker.utils")
    if utils is not None and not utils.in_toplevel_process():
        return "compile worker"
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and "compile_worker" in (spec.name or ""):
        return "compile pool"
    mp = sys.modules.get("multiprocessing")
    if mp is not None and mp.parent_process() is not None:
        return mp.current_process().name
    return "main"


def _metadata_events(pid: int) -> list[dict]:
    return [
        {"name": "process_name", "ph": "M", "pid": pid, "args": {"name": f"{_role()} {pid}"}},
        {
            "name": "thread_name", "ph": "M", "pid": pid, "tid": pid | _SUBPROCESS_TID_BIT, "args":
            {"name": "subprocess"}
        },
    ]


@_guarded
def flush() -> None:
    global _named_pid
    pid = os.getpid()
    with _lock:
        events = _events[:]
        _events.clear()
    if not events:
        return
    if _named_pid != pid:
        # Named at the first flush, by when Inductor has marked its workers.
        events = _metadata_events(pid) + events
        _named_pid = pid
    data = "".join(json.dumps(e, separators=(",", ":")) + ",\n" for e in events).encode()
    fd = os.open(_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _after_fork_in_child() -> None:
    global _lock, _named_pid
    # Another thread may have held the lock at fork time.
    _lock = threading.Lock()
    _events.clear()
    _load_starts.clear()
    _named_pid = None


# Stages


def _add_native(trace_json: str, base_us: float) -> None:
    tid = threading.get_native_id()
    for e in json.loads(trace_json).get("traceEvents", []):
        # "Total ..." events are per-name aggregates on their own fake threads.
        if e.get("ph") != "X" or e["name"].startswith("Total "):
            continue
        args = e.get("args", {})
        name = e["name"]
        if name in ("RunPass", "OptFunction", "OptModule") and "detail" in args:
            name = args["detail"]
        _span(name, base_us + e["ts"], base_us + e["ts"] + e["dur"], "native", args, tid=tid)


def _stage_function(name: str, fn) -> str:
    """The function a stage runs, e.g. CUDABackend.make_ttgir.

    Backends register stages as lambdas calling a backend method; the method is
    the first name the lambda looks up that is callable on its `self`.
    """
    if getattr(fn, "__name__", "") != "<lambda>":
        return getattr(fn, "__qualname__", name)
    cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
    if "self" in cells:
        owner = cells["self"].cell_contents
        for attr in fn.__code__.co_names:
            if callable(getattr(owner, attr, None)):
                return f"{type(owner).__name__}.{attr}"
    return name


@_guarded
def _stage_begin() -> tuple[float, int]:
    from triton._C.libtriton import llvm

    start = _now_us()
    return start, llvm.time_trace_begin(_GRANULARITY_US) if hasattr(llvm, "time_trace_begin") else -1


@_guarded
def _stage_end(name: str, function: str, metadata: dict, start: float, base_ns: int, trace_json: str) -> None:
    end = _now_us()
    if getattr(_compile, "first_start", None) is None:
        _compile.first_start = start
    _compile.last_end = end
    _span(name, start, end, "stage", {"function": function, "kernel": metadata.get("name", "")})
    if trace_json:
        _add_native(trace_json, base_ns / 1000)
    _memory_counter(end)


def _wrap_stage(name: str, fn):
    function = _stage_function(name, fn)

    def stage(src, metadata):
        begin = _stage_begin()
        try:
            return fn(src, metadata)
        finally:
            if begin is not None:
                trace_json = ""
                if begin[1] >= 0:
                    from triton._C.libtriton import llvm

                    # Stopped unconditionally once started, even if tracing was
                    # disabled meanwhile, so the profiler never stays on.
                    trace_json = llvm.time_trace_end()
                _stage_end(name, function, metadata, *begin, trace_json)

    return stage


@_guarded
def _mark(event: str) -> None:
    setattr(_compile, event, _now_us())


@_guarded
def _wrap_stages(stages: dict) -> None:
    _mark("stages_added")
    for name, fn in list(stages.items()):
        stages[name] = _wrap_stage(name, fn)


# Compile and load events


@_guarded
def _record_compile(src, metadata: dict, times, cache_hit: bool, end: float) -> None:
    from triton import knobs
    from triton.runtime.cache import FileCacheManager

    marks = {k: getattr(_compile, k, None) for k in ("key_done", "stages_added", "first_start", "last_end")}
    for k in marks:
        setattr(_compile, k, None)
    name = metadata.get("name") or getattr(src, "name", "")
    manager = (knobs.cache.manager_class or FileCacheManager).__name__
    args = {
        "function": "triton.compiler.compile", "kernel": name, "hash": metadata.get("hash", ""), "cache_hit": cache_hit
    }
    if cache_hit or marks["first_start"] is None:
        start = end - times.total
        _span(f"compile {name}" + (" (cache hit)" if cache_hit else ""), start, end, "compile", args)
        steps = [("cache_key", "triton.compiler.compiler.get_cache_key", marks["key_done"]),
                 ("cache_lookup", f"{manager}.get_group", end)]
    else:
        # CompileTimes leaves gaps untimed, so anchor on the stage spans rather
        # than subtracting durations from the end.
        start = marks["first_start"] - times.ir_initialization
        _span(f"compile {name}", start, end, "compile", args)
        # AST -> TTIR; an IRSource parses an IR file instead.
        frontend = "ast" if type(src).__name__ == "ASTSource" else "parse_ir"
        steps = [("cache_key", "triton.compiler.compiler.get_cache_key", marks["key_done"]),
                 ("cache_lookup", f"{manager}.get_group", marks["stages_added"]),
                 (frontend, f"{type(src).__name__}.make_ir", marks["first_start"])]
        _span("cache_store", marks["last_end"], end, "stage", {"function": f"{manager}.put", "kernel": name})
    # Steps whose boundary wasn't observed in this compile are skipped.
    step_start = start
    for step, function, step_end in steps:
        if step_end is not None and step_start <= step_end <= end:
            _span(step, step_start, step_end, "stage", {"function": function, "kernel": name})
            step_start = step_end


@_guarded
def _record_listener(listener, start: float, end: float) -> None:
    function = f"{getattr(listener, '__module__', '')}.{getattr(listener, '__qualname__', repr(listener))}"
    _span("compilation listener", start, end, "listener", {"function": function})


@_guarded
def _load_start(module, function, name, metadata_group, hash) -> None:
    _load_starts[hash] = _now_us()


@_guarded
def _load_end(module, function, name, metadata_group, hash) -> None:
    start = _load_starts.pop(hash, None)
    if start is not None:
        args = {"function": "CompiledKernel._init_handles", "kernel": name, "hash": hash}
        _span(f"load {name}", start, _now_us(), "load", args)
        flush()


def _install_hooks() -> None:
    from triton import knobs

    # Every hook is looked up before any is installed, so a Triton build
    # missing one raises here without leaving the others half installed.
    # Hooks already installed by others are still called, unguarded, so their
    # behavior is unchanged.
    prev_stages_hook = knobs.runtime.add_stages_inspection_hook
    prev_listener = knobs.compilation.listener
    load_start_hooks = knobs.runtime.kernel_load_start_hook
    load_end_hooks = knobs.runtime.kernel_load_end_hook

    def stages_hook(*args, **kwargs):
        if not args and not kwargs:
            # compile() and the JIT call it bare for a cache-key suffix. An
            # empty suffix keeps traced compiles on the same cache entries.
            # In compile() this call directly follows get_cache_key.
            _mark("key_done")
            return prev_stages_hook() if prev_stages_hook else ("", "")
        if prev_stages_hook:
            prev_stages_hook(*args, **kwargs)
        _wrap_stages(next(a for a in args if isinstance(a, dict)))

    def listener(*, src, metadata, metadata_group, times, cache_hit):
        # The compile ends here; a previous listener's time is its own span.
        end = _now_us()
        _record_compile(src, metadata, times, cache_hit, end)
        if prev_listener:
            prev_listener(src=src, metadata=metadata, metadata_group=metadata_group, times=times, cache_hit=cache_hit)
            _record_listener(prev_listener, end, _now_us())
        flush()

    knobs.runtime.add_stages_inspection_hook = stages_hook
    knobs.compilation.listener = listener
    load_start_hooks.add(_load_start)
    load_end_hooks.add(_load_end)


def _resolve_path(path: str) -> str:
    if inherited := os.environ.get(_FILE_ENV):
        return inherited
    if path.endswith(os.sep) or os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return os.path.join(path, f"triton_compile_trace_{stamp}_{os.getpid()}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def _give_up(message: str) -> None:
    warnings.warn(f"TRITON_COMPILE_TRACE: {message}, tracing disabled")
    # Processes started from here would fail the same way, each warning again.
    os.environ.pop("TRITON_COMPILE_TRACE", None)
    os.environ.pop(_FILE_ENV, None)


def enable(path: str) -> None:
    global _path
    if _path is not None:
        return
    try:
        path = _resolve_path(path)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        os.write(fd, b"[\n")
        os.close(fd)
    except FileExistsError:
        pass
    except OSError as e:
        _give_up(f"cannot create {path}: {e!r}")
        return
    try:
        _install_hooks()
    except Exception as e:
        _give_up(f"cannot install hooks: {e!r}")
        return
    _path = path
    # Exported only once tracing works, for the processes this one starts.
    os.environ[_FILE_ENV] = path
    atexit.register(flush)
    os.register_at_fork(after_in_child=_after_fork_in_child)


# Subprocesses


def _sample_child(pid: int, stop: threading.Event, samples: list) -> None:
    own_exe = os.readlink("/proc/self/exe")
    while not stop.is_set():
        try:
            # Until exec the child still runs our image and reports our memory.
            if os.readlink(f"/proc/{pid}/exe") != own_exe:
                with open(f"/proc/{pid}/status") as f:
                    fields = dict(line.split(":", 1) for line in f)
                samples.append(
                    (_now_us(), int(fields["VmRSS"].split()[0]) / 1024, int(fields["VmHWM"].split()[0]) / 1024))
        except (OSError, KeyError, ValueError):
            # The child exited between checks.
            break
        stop.wait(_SAMPLE_S)


@_guarded
def _observe_begin(pid: int, start: float) -> dict:
    global _subprocess_events
    with _lock:
        _subprocess_events += 1
        seq = _subprocess_events
    obs = {
        "caller": sys._getframe(3).f_code.co_qualname,
        "start": start,
        "seq": seq,
        "children_cpu": resource.getrusage(resource.RUSAGE_CHILDREN),
        "stop": threading.Event(),
        "samples": [],
    }
    obs["sampler"] = threading.Thread(target=_sample_child, args=(pid, obs["stop"], obs["samples"]), daemon=True)
    obs["sampler"].start()
    return obs


@_guarded
def _observe_end(cmd: list[str], proc: subprocess.Popen, obs: dict) -> None:
    global _subprocess_events
    end = _now_us()
    obs["sampler"].join()
    with _lock:
        _subprocess_events += 1
        alone = _subprocess_events == obs["seq"] + 1
    args = {"function": obs["caller"], "pid": proc.pid, "returncode": proc.returncode, "cmd": " ".join(cmd)}
    # Children reaped automatically (SIGCHLD ignored) are not in RUSAGE_CHILDREN.
    if alone and signal.getsignal(signal.SIGCHLD) != signal.SIG_IGN:
        before, after = obs["children_cpu"], resource.getrusage(resource.RUSAGE_CHILDREN)
        args["cpu_s"] = round(after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime, 3)
    args["peak_rss_mb"] = round(max((hwm for _, _, hwm in obs["samples"]), default=0.0), 1)
    tool = os.path.basename(cmd[0])
    _span(tool, obs["start"], end, "subprocess", args, tid=os.getpid() | _SUBPROCESS_TID_BIT)
    for ts, rss, _ in obs["samples"]:
        _counter(f"{tool} RSS (MB)", ts, {"rss": round(rss, 1)})
    _counter(f"{tool} RSS (MB)", end, {"rss": 0})


def _traceable(kwargs: dict) -> bool:
    if any(k in kwargs for k in ("input", "timeout", "capture_output")):
        return False
    return subprocess.PIPE not in (kwargs.get("stdout"), kwargs.get("stderr"))


def run_subprocess(cmd: list[str], *, check: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run for compiler tools such as ptxas, traced when enabled.

    The child is run and waited on as subprocess.run does, so tracing only
    observes it and cannot change the outcome. Peak RSS comes from sampling the
    child's VmHWM: its own rusage would include the parent's peak, as it is
    spawned with vfork. CPU time is the RUSAGE_CHILDREN delta, reported only
    when no other traced subprocess of this process started or ended meanwhile
    and SIGCHLD isn't ignored.
    Calls that capture output are not traced.
    """
    if _path is None or not _traceable(kwargs):
        return subprocess.run(cmd, check=check, **kwargs)
    start = _now_us()
    with subprocess.Popen(cmd, **kwargs) as proc:
        obs = _observe_begin(proc.pid, start)
        try:
            proc.wait()
        except BaseException:
            proc.kill()
            raise
        finally:
            if obs is not None:
                obs["stop"].set()
                _observe_end(cmd, proc, obs)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return subprocess.CompletedProcess(cmd, proc.returncode)
