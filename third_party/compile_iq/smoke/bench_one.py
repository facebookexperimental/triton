"""Base-env (Python 3.13) worker for the EVO cross-env bridge.

Compiles + benchmarks ONE ptxas ACF candidate and prints its score. The EVO engine
lives in a separate 3.10 conda env that has no triton/torch, so it cannot benchmark a
Triton kernel itself; its objective shells out to this worker (see evo_search.py).

Reads the EVO candidate (a hex-encoded ACF string) from a file, materializes a real
ptxas ACF (the ACF is just the hex decoded to bytes -- same encoding EVO and CIQ share),
then benchmarks the collected kernel with the ACF applied via compile_iq's replay path.

Usage: python bench_one.py <task_dir> <acf_hex_file> <per_candidate_timeout_s>
Prints exactly one line: "MS <float>" on success, or "INVALID" (diverged/wedged/crashed).
"""
import os
import sys
import tempfile

from triton.compile_iq.factory import _isolated_bench


def main():
    task_dir, acf_hex_file, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(acf_hex_file) as f:
        acf = f.read().strip()
    with tempfile.NamedTemporaryFile(suffix=".acf", delete=False) as t:
        t.write(bytes.fromhex(acf))
        acf_path = t.name
    try:
        ms = _isolated_bench(task_dir, acf_path, timeout)
    finally:
        os.unlink(acf_path)
    print(f"MS {ms}" if isinstance(ms, (int, float)) else "INVALID")


if __name__ == "__main__":
    main()
