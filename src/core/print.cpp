#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <type_traits>

#include "flashgrad/core/tensor.h"

namespace flashgrad {

namespace {

constexpr std::int64_t kSummarizeThreshold = 1000;  // elements
constexpr std::int64_t kEdgeItems = 3;              // shown at each end when summarizing
constexpr std::size_t kIndent = 7;                  // strlen("tensor(")

template <typename T>
void print_scalar(std::ostream& os, T value) {
  if constexpr (std::is_same_v<T, bool>) {
    os << (value ? "true" : "false");
  } else {
    os << value;
  }
}

template <typename T>
void print_values(std::ostream& os, const T* data, const Shape& shape, const Strides& strides,
                  std::size_t dim, bool summarize) {
  if (dim == shape.size()) {
    print_scalar(os, *data);
    return;
  }
  // Rows of a matrix go on separate lines, matrices are separated by an empty line, etc.
  const bool innermost = dim + 1 == shape.size();
  const std::string separator = innermost ? ", "
                                          : "," + std::string(shape.size() - dim - 1, '\n') +
                                                std::string(kIndent + dim + 1, ' ');
  const std::int64_t n = shape[dim];
  os << '[';
  for (std::int64_t i = 0; i < n; ++i) {
    if (i > 0) os << separator;
    if (summarize && n > 2 * kEdgeItems && i == kEdgeItems) {
      os << "..." << separator;
      i = n - kEdgeItems;
    }
    print_values(os, data + i * strides[dim], shape, strides, dim + 1, summarize);
  }
  os << ']';
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
  if (!t.defined()) return os << "tensor(undefined)";
  const Tensor host = t.cpu();
  os << "tensor(";
  if (host.numel() == 0) {
    os << "[], shape=" << to_string(host.shape());
  } else {
    dispatch_all(host.dtype(), "print", [&]<typename T>() {
      print_values(os, host.data_ptr<T>(), host.shape(), host.strides(), 0,
                   host.numel() > kSummarizeThreshold);
    });
  }
  return os << ", dtype=" << t.dtype() << ", device=" << t.device() << ")";
}

}  // namespace flashgrad
