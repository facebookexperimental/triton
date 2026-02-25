#ifndef PROTON_UTILITY_ENV_H_
#define PROTON_UTILITY_ENV_H_

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>

inline bool getBoolEnv(const std::string &env, bool defaultValue) {
  static std::mutex getenv_mutex;
  std::lock_guard<std::mutex> lock(getenv_mutex);
  const char *s = std::getenv(env.c_str());
  if (s == nullptr)
    return defaultValue;
  std::string str(s);
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "on" || str == "true" || str == "1";
}

#endif // PROTON_UTILITY_ENV_H_
