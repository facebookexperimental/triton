"""triton.language.extra.tlx — re-export from tlx_plugin.

This package is symlinked into triton/python/triton/language/extra/tlx
by TLXPlugin's setup.sh so that `import triton.language.extra.tlx as tlx`
resolves to the TLXPlugin's Python DSL.
"""

import os as _os
import sys as _sys

# Ensure the TLXPlugin python dir is on sys.path so tlx_plugin is importable
_plugin_python_dir = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))
if _plugin_python_dir not in _sys.path:
    _sys.path.insert(0, _plugin_python_dir)

from tlx_plugin import *  # noqa: F401,F403
from tlx_plugin import __all__  # noqa: F401
