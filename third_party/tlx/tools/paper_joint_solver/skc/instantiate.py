"""SkeletonInstantiator — render a runnable kernel module (SKC step 3).

The instance file is standalone-importable (bench loads modules by path):
it inserts the skc package dir on sys.path, imports the skeleton, and pins
the bound parameters.  The full binding audit is embedded as a JSON header
so every instance carries its own provenance.
"""

import json
from pathlib import Path

_TEMPLATE = '''"""SKC instance — generated, do not edit.

Solution : {solution}
DDG      : {ddg}
Skeleton : skc.{skeleton_mod} (verified handwritten-kernel protocol)

Binding audit:
{audit_block}
"""

import sys
from pathlib import Path

sys.path.insert(0, {pkg_dir!r})

from skc.{skeleton_mod} import {entry} as _skeleton_entry  # noqa: E402

PARAMS = {params}

AUDIT = {audit}


def {entry}(*args, **kwargs):
    return _skeleton_entry(*args, params=PARAMS, **kwargs)
'''

_SKELETONS = {
    "fwd": ("skeleton_fwd", "attention"),
    "bwd": ("skeleton_bwd", "bwd_attention"),
}


def render(params: dict, audit: dict, *, solution_path, ddg_path, out_path,
           skeleton="fwd"):
    skeleton_mod, entry = _SKELETONS[skeleton]
    pkg_dir = str(Path(__file__).resolve().parent.parent)
    audit_json = json.dumps(audit, indent=2, default=str)
    text = _TEMPLATE.format(
        solution=solution_path,
        ddg=ddg_path,
        skeleton_mod=skeleton_mod,
        entry=entry,
        audit_block="\n".join("  " + ln for ln in audit_json.splitlines()),
        pkg_dir=pkg_dir,
        params=json.dumps(params, indent=4),
        audit=audit_json,
    )
    Path(out_path).write_text(text)
    return out_path
