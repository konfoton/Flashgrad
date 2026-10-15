#pragma once

#include <cstdint>

#include "flashgrad/core/error.h"
#include "flashgrad/core/shape.h"
#include "flashgrad/core/tensor.h"

// Argument validation shared by the tensor implementation.
namespace flashgrad::detail {

inline void check_defined(const char* op, const Tensor& t) {
  FG_CHECK(t.defined(), op, ": tensor is undefined");
}

// Rejects writes into tensors where several elements share one memory location
// (expanded dimensions), since the result would depend on the write order.
inline void check_writable(const char* op, const Tensor& t) {
  for (std::int64_t d = 0; d < t.dim(); ++d) {
    FG_CHECK(t.stride(d) != 0 || t.size(d) <= 1, op,
             ": cannot write into a tensor whose elements overlap (expanded dimension ", d,
             "), got ", describe(t), " with strides ", to_string(t.strides()));
  }
}

}  // namespace flashgrad::detail
