#include "flashgrad/core/error.h"

namespace flashgrad {

namespace {

std::string format_what(const std::string& message, const char* file, int line) {
  return detail::concat(message, "\n  at ", file, ":", line);
}

}  // namespace

Error::Error(std::string message, const char* file, int line)
    : std::runtime_error(format_what(message, file, line)),
      message_(std::move(message)),
      file_(file),
      line_(line) {}

namespace detail {

void throw_error(const char* file, int line, const char* condition,
                 const std::string& message) {
  if (condition == nullptr) {
    throw Error(message, file, line);
  }
  if (message.empty()) {
    throw Error(concat("Check failed: ", condition), file, line);
  }
  throw Error(concat("Check failed: ", condition, " — ", message), file, line);
}

}  // namespace detail
}  // namespace flashgrad
