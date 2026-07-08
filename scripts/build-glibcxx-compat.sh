#!/usr/bin/env bash
#
# Build an augmented "libstdc++.so.6" that adds the GLIBCXX_3.4.30 (GCC 12)
# symbols the prebuilt LLVM tools reference, on a host whose system libstdc++ is
# older (e.g. glibc-toolset / EL9 ships libstdc++ 11 == GLIBCXX_3.4.29).
#
# Why this is needed
# ------------------
# Triton's default prebuilt LLVM (e.g. the "ubuntu-x64" image) is compiled on a
# newer distro. Its build-time tools (mlir-tblgen, opt, ...) are run during the
# Triton build and dynamically require symbols versioned GLIBCXX_3.4.30, which
# the host's libstdc++.so.6 does not provide:
#     mlir-tblgen: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
#
# glibc's version check is bound to the *soname* "libstdc++.so.6", so an
# LD_PRELOADed shim under a different name does NOT satisfy it -- the loaded
# libstdc++.so.6 itself must carry the version node. We therefore build a drop-in
# libstdc++.so.6 that:
#   * keeps the soname "libstdc++.so.6";
#   * declares every version node the real lib defines (so all older
#     GLIBCXX_/CXXABI_ requirements still resolve) plus GLIBCXX_3.4.30;
#   * defines the two GLIBCXX_3.4.30 symbols missing from the base lib:
#       - std::__glibcxx_assert_fail  (new in GCC 12; implemented as a trap)
#       - std::condition_variable::wait(unique_lock&)  (only re-versioned in
#         GCC 12; forwarded to the base lib's GLIBCXX_3.4.11 implementation);
#   * DT_NEEDEDs the real lib (copied under the renamed soname
#     "libstdc++_base.so.6") so every other symbol resolves against it.
#
# The result is added to LD_LIBRARY_PATH for the build only (see setup.py,
# TRITON_GLIBC_COMPAT). It is a build-host workaround and is not shipped.
#
# Usage: build-glibcxx-compat.sh <output_dir>
set -euo pipefail

OUT_DIR="${1:?usage: build-glibcxx-compat.sh <output_dir>}"
mkdir -p "$OUT_DIR"

# Locate the real system libstdc++.so.6.
REAL="$(gcc -print-file-name=libstdc++.so.6 2>/dev/null || true)"
if [ -z "$REAL" ] || [ ! -e "$REAL" ]; then
    REAL="/lib64/libstdc++.so.6"
fi
REAL="$(readlink -f "$REAL")"
if [ ! -e "$REAL" ]; then
    echo "error: could not find system libstdc++.so.6" >&2
    exit 1
fi

BASE="$OUT_DIR/libstdc++_base.so.6"
AUG="$OUT_DIR/libstdc++.so.6"
STAMP="$OUT_DIR/.glibcxx-compat.stamp"

# Skip if already built against the same base lib.
if [ -f "$STAMP" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$REAL" ] \
        && [ -e "$AUG" ] && [ "$AUG" -nt "$REAL" ]; then
    echo "glibcxx-compat: up to date ($AUG)"
    exit 0
fi

echo "glibcxx-compat: real libstdc++ = $REAL"

# 1) Copy the real lib and rename its soname so it can co-exist with our lib.
cp -f "$REAL" "$BASE"
chmod u+w "$BASE"
patchelf --set-soname libstdc++_base.so.6 "$BASE"

# 2) Generate a source + version script: a dummy symbol per existing version
#    node (forces ld to emit each Verdef, satisfying the loader's version check)
#    plus the two GLIBCXX_3.4.30 symbols the base lib lacks.
SHIM_C="$OUT_DIR/glibcxx_compat_shim.c"
SHIM_MAP="$OUT_DIR/glibcxx_compat_shim.map"
: > "$SHIM_C"
: > "$SHIM_MAP"

i=0
while read -r v; do
    [ -n "$v" ] || continue
    i=$((i + 1))
    printf 'void __triton_cxxcompat_%s(void){}\n' "$i" >> "$SHIM_C"
    printf '%s {\n  global:\n    __triton_cxxcompat_%s;\n};\n' "$v" "$i" >> "$SHIM_MAP"
done < <(objdump -T "$REAL" | grep -oE '(GLIBCXX|CXXABI)_[0-9.]+[0-9]' | sort -u -V)

cat >> "$SHIM_C" <<'EOF'
/* GLIBCXX_3.4.30 additions absent from the base (<= 3.4.29) libstdc++. */

/* New in GCC 12: std::__glibcxx_assert_fail(const char*,int,const char*,const char*).
 * Only invoked when a libstdc++ hardening assertion fails; trap if it ever is. */
void _ZSt21__glibcxx_assert_failPKciS0_S0_(const char *file, int line,
                                           const char *func, const char *cond) {
  (void)file; (void)line; (void)func; (void)cond;
  __builtin_trap();
}

/* Only re-versioned in GCC 12: std::condition_variable::wait(unique_lock<mutex>&).
 * Forward to the base lib's GLIBCXX_3.4.11 implementation (ABI-compatible). */
__asm__(".symver __triton_cv_wait_base,"
        "_ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE@GLIBCXX_3.4.11");
extern void __triton_cv_wait_base(void *cv, void *lock);
void _ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE(void *cv, void *lock) {
  __triton_cv_wait_base(cv, lock);
}
EOF

cat >> "$SHIM_MAP" <<'EOF'
GLIBCXX_3.4.30 {
  global:
    _ZSt21__glibcxx_assert_failPKciS0_S0_;
    _ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE;
};
EOF

# 3) Link the drop-in libstdc++.so.6: same soname, chained to the renamed base.
gcc -shared -fPIC -O2 -o "$AUG" "$SHIM_C" \
    -Wl,--version-script="$SHIM_MAP" \
    -Wl,-soname,libstdc++.so.6 \
    -L"$OUT_DIR" -l:libstdc++_base.so.6 \
    -Wl,-rpath,'$ORIGIN'

echo "$REAL" > "$STAMP"
echo "glibcxx-compat: built $AUG"