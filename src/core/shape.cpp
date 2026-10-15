#include "flashgrad/core/shape.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>

#include "flashgrad/core/error.h"

namespace flashgrad {

std::int64_t numel(const Shape& shape) {
  std::int64_t n = 1;
  for (const std::int64_t size : shape) {
    FG_CHECK(size >= 0, "numel: negative dimension in shape ", to_string(shape));
    FG_CHECK(size == 0 || n <= std::numeric_limits<std::int64_t>::max() / size, "numel: shape ",
             to_string(shape), " has too many elements");
    n *= size;
  }
  return n;
}

Strides contiguous_strides(const Shape& shape) {
  Strides strides(shape.size());
  std::int64_t stride = 1;
  for (std::size_t d = shape.size(); d-- > 0;) {
    strides[d] = stride;
    // Keep strides non-zero for empty dimensions (as PyTorch does).
    stride *= std::max<std::int64_t>(shape[d], 1);
  }
  return strides;
}

Shape broadcast_shapes(const Shape& a, const Shape& b, const char* op) {
  const std::size_t ndim = std::max(a.size(), b.size());
  const std::size_t pad_a = ndim - a.size();
  const std::size_t pad_b = ndim - b.size();
  Shape out(ndim);
  for (std::size_t d = 0; d < ndim; ++d) {
    const std::int64_t size_a = d < pad_a ? 1 : a[d - pad_a];
    const std::int64_t size_b = d < pad_b ? 1 : b[d - pad_b];
    FG_CHECK(size_a == size_b || size_a == 1 || size_b == 1, op, ": shapes ", to_string(a), " and ",
             to_string(b), " are not broadcastable");
    out[d] = size_a == 1 ? size_b : size_a;
  }
  return out;
}

std::int64_t normalize_dim(std::int64_t dim, std::int64_t ndim, const char* op) {
  FG_CHECK(ndim > 0, op, ": dimension ", dim, " given for a 0-dimensional tensor");
  FG_CHECK(dim >= -ndim && dim < ndim, op, ": dimension ", dim, " out of range for a ", ndim,
           "-D tensor (expected [", -ndim, ", ", ndim - 1, "])");
  return dim < 0 ? dim + ndim : dim;
}

std::string to_string(const Shape& shape) {
  std::ostringstream oss;
  oss << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << shape[i];
  }
  oss << ']';
  return oss.str();
}

}  // namespace flashgrad
