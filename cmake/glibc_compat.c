/*
 * glibc 2.38+ -> 2.34 compatibility trampolines.
 *
 * The default prebuilt LLVM image Triton links against (e.g. the "ubuntu-x64"
 * build selected in python/build_helpers.py) is compiled on a host with
 * glibc >= 2.38. Since glibc 2.38, <stdlib.h>/<stdio.h> redirect the strtol()
 * and scanf() families to __isoc23_* variants (they differ only in accepting
 * C23 integer literals such as 0b1010). Those __isoc23_* symbols do NOT exist
 * on older glibc, so linking libtriton against that LLVM leaves undefined
 * "__isoc23_*@GLIBC_2.38" references that cannot be resolved on a host whose
 * glibc predates 2.38 -- the build/import fails with a missing GLIBC_2.38
 * symbol.
 *
 * Each trampoline below defines the missing __isoc23_* symbol locally and
 * forwards to its pre-C23 equivalent, which is ABI-compatible and present on
 * glibc 2.34. Because this file is compiled on the (old-glibc) build host, its
 * own calls to strtol()/vsscanf()/... are NOT redirected (the redirect macros
 * only exist in glibc >= 2.38 headers), so they bind to the real glibc 2.34
 * symbols.
 *
 * This TU is only compiled when the TRITON_GLIBC_COMPAT CMake option is ON,
 * which should only be enabled when building against an LLVM compiled on a
 * newer glibc than the build/run host. The symbols keep default visibility so
 * they also satisfy references from any separately loaded LLVM shared object or
 * backend plugin, not just libtriton's own statically linked LLVM archives.
 *
 * NOTE: this list is the superset that the CUDA/LLVM static archives are known
 * to pull in (see D110981848). If a build still shows an undefined __isoc23_*
 * symbol
 *     objdump -T libtriton.so | grep -E 'UND .*GLIBC_2\.3[89]'
 * add the corresponding trampoline here and rebuild.
 */

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* strtol() family: (const char *nptr, char **endptr, int base) */
#define TLX_TRAMPOLINE_STRTO(name, ret_type)                                   \
  ret_type __isoc23_##name(const char *nptr, char **endptr, int base) {        \
    return name(nptr, endptr, base);                                           \
  }

TLX_TRAMPOLINE_STRTO(strtol, long)
TLX_TRAMPOLINE_STRTO(strtoll, long long)
TLX_TRAMPOLINE_STRTO(strtoul, unsigned long)
TLX_TRAMPOLINE_STRTO(strtoull, unsigned long long)
TLX_TRAMPOLINE_STRTO(strtoimax, intmax_t)
TLX_TRAMPOLINE_STRTO(strtoumax, uintmax_t)

#undef TLX_TRAMPOLINE_STRTO

/* scanf() family: variadic entry points forward through the v* variants. */
int __isoc23_scanf(const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  int r = vscanf(format, ap);
  va_end(ap);
  return r;
}

int __isoc23_fscanf(FILE *stream, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  int r = vfscanf(stream, format, ap);
  va_end(ap);
  return r;
}

int __isoc23_sscanf(const char *str, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  int r = vsscanf(str, format, ap);
  va_end(ap);
  return r;
}

int __isoc23_vscanf(const char *format, va_list ap) {
  return vscanf(format, ap);
}

int __isoc23_vfscanf(FILE *stream, const char *format, va_list ap) {
  return vfscanf(stream, format, ap);
}

int __isoc23_vsscanf(const char *str, const char *format, va_list ap) {
  return vsscanf(str, format, ap);
}
