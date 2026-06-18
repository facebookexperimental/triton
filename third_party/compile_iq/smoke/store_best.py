"""Base-env (Python 3.13) store step for the EVO factory route.

Persists the best ACF found by the EVO search (evo_search.py) into the CIQ ACF store,
keyed by the collected kernel's PTX hash + arch, so the consume hook HITs on the next run.
This is the EVO equivalent of the store call CIQ's own factory makes.

Usage: python store_best.py <task_dir> <best_acf_hex_file>
"""
import json
import os
import sys

from triton.compile_iq import store


def main():
    task_dir, best_hex_file = sys.argv[1], sys.argv[2]
    with open(os.path.join(task_dir, "task.json")) as f:
        task = json.load(f)
    with open(best_hex_file) as f:
        acf_hex = f.read().strip()

    # The ACF is just the hex candidate decoded to bytes (no extra framing).
    acf_bytes = bytes.fromhex(acf_hex)

    meta = {"ptx_sha256": task["ptx_sha256"], "arch": task["arch"], "source": "evo", "smoke_test": True}
    p = store.write_acf(task["ptx_sha256"], task["arch"], acf_bytes, meta)
    print(f"[store-best] wrote ACF ({len(acf_bytes)} bytes) -> {p}")


if __name__ == "__main__":
    main()
