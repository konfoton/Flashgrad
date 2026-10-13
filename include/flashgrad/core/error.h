#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace flashgrad {

// Base exception for every error raised by the library.
class Error : public std::runtime_error {
 public:
  Error(std::string message, const char* file, int line);

  const std::string& message() const noexcept { return message_; }
  const char* file() const noexcept { return file_; }
  int line() const noexcept { return line_; }

 private:
  std::string message_;
  const char* file_;
  int line_;
};

namespace detail {

template <typename... Args>
std::string concat(Args&&... args) {
  if constexpr (sizeof...(Args) == 0) {
    return {};
  } else {
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    return oss.str();
  }
}

[[noreturn]] void throw_error(const char* file, int line, const char* condition,
                              const std::string& message);

}  // namespace detail
}  // namespace flashgrad

// FG_CHECK(cond, msg...) — throws flashgrad::Error when `cond` is false.
// Message parts are streamed, so any type with operator<< can be used:
//   FG_CHECK(a.dim() == 2, "matmul: expected 2D tensor, got ", a.dim(), "D");
#define FG_CHECK(cond, ...)                                                    \
  do {                                                                         \
    if (!(cond)) [[unlikely]] {                                                \
      ::flashgrad::detail::throw_error(__FILE__, __LINE__, #cond,              \
                                       ::flashgrad::detail::concat(__VA_ARGS__)); \
    }                                                                          \
  } while (0)

// FG_THROW(msg...) — unconditionally throws flashgrad::Error.
#define FG_THROW(...)                                                          \
  ::flashgrad::detail::throw_error(__FILE__, __LINE__, nullptr,                \
                                   ::flashgrad::detail::concat(__VA_ARGS__))
