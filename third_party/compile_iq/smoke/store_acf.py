"""Store an EVO-minted ACF into the CIQ ACF store, keyed by a COLLECTED task's spec
(sha256(normalized PTX) x arch), so the consume hook HITs on the next run.

This is the PTX-direct route's persist step (the EVO orchestrator only writes best.acf.hex; this
puts it in the store the make_cubin consume hook reads). Usage: python store_acf.py <task_dir> <hex>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the compile_iq pkg root
from compile_iq import store


def main():
    # TODO(compile_iq perf, item 3): gate stores on a heavy locked-clock A/B re-validation of the
    # candidate (the ws_ab.py win-gate on branch daohang/compile_iq_perf_harness) -- only persist ACFs
    # that beat the no-ACF baseline beyond the noise floor, so a winner's-curse / noise "win" is never
    # stored. Today this stores the search's best unconditionally.
    task_dir, hex_file = sys.argv[1], sys.argv[2]
    spec = json.load(open(os.path.join(task_dir, "spec.json")))
    acf = bytes.fromhex(open(hex_file).read().strip())
    meta = {"ptx_sha256": spec["ptx_sha256"], "arch": spec["arch"], "entry": spec.get("entry"), "source": "ptx_evo"}
    p = store.write_acf(spec["ptx_sha256"], spec["arch"], acf, meta)
    print(f"[store-acf] wrote ACF ({len(acf)} bytes) for {spec.get('entry')} -> {p}")


if __name__ == "__main__":
    main()
