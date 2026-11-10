"""compile_iq switches, parsed like any other Triton boolean env var ("1"/"true"/"on" enable,
"0"/"false"/"off" disable). Callers outside Triton (e.g. Inductor, which must turn off its static
launcher and suppress candidate benchmarking while collection is on) use these too, so both sides
agree on when compile_iq is enabled.
"""

from triton._C.libtriton import getenv_bool


def collect_enabled() -> bool:
    return getenv_bool("TRITON_COMPILE_IQ_COLLECT", False)


def apply_enabled() -> bool:
    return getenv_bool("TRITON_COMPILE_IQ_APPLY", False)
