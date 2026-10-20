#pragma once

#include <cstdint>

#include "flashgrad/autograd/grad_mode.h"
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

// In-place ops aren't recorded, so with grad mode on they may neither change a tensor that
// requires grad (backward may need its old values) nor read from one (the result would
// silently lose its history).
inline void check_in_place(const char* op, const Tensor& self, const Tensor* source = nullptr) {
  if (!is_grad_enabled()) return;
  FG_CHECK(!self.requires_grad(), op, ": cannot modify ", describe(self),
           " in place because it requires grad; update parameters inside a NoGradGuard");
  FG_CHECK(source == nullptr || !source->requires_grad(), op, ": the source ", describe(*source),
           " requires grad, but in-place ops aren't recorded; detach() it or use a NoGradGuard");
}

}  // namespace flashgrad::detail
