#ifndef TRITON_UNITTEST_UTIL_SCOPEDALLOWNPOT_H
#define TRITON_UNITTEST_UTIL_SCOPEDALLOWNPOT_H

#include <cstdlib>
#include <optional>
#include <string>

namespace mlir::triton {

// Sets TRITON_ALLOW_NPOT for one scope ("1" by default, nullptr unsets it) and
// restores the previous value. Flag-off LinearLayout rejects NPOT out-dims.
class ScopedAllowNpot {
public:
  explicit ScopedAllowNpot(const char *value = "1") {
    if (const char *previous = std::getenv(kVar))
      saved = std::string(previous);
    apply(value);
  }
  ~ScopedAllowNpot() { apply(saved ? saved->c_str() : nullptr); }
  ScopedAllowNpot(const ScopedAllowNpot &) = delete;
  ScopedAllowNpot &operator=(const ScopedAllowNpot &) = delete;

private:
  static constexpr const char *kVar = "TRITON_ALLOW_NPOT";
  static void apply(const char *value) {
    if (value)
      setenv(kVar, value, /*overwrite=*/1);
    else
      unsetenv(kVar);
  }
  std::optional<std::string> saved;
};

} // namespace mlir::triton

#endif // TRITON_UNITTEST_UTIL_SCOPEDALLOWNPOT_H
