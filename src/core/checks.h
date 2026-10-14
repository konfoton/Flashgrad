#pragma once

#include <cstdint>

#include "flashgrad/core/device.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/tensor.h"

// Argument validation shared by the core operations.
namespace flashgrad::detail {

inline void check_defined(const char* op, const Tensor& t) {
  FG_CHECK(t.defined(), op, ": tensor is undefined");
}

inline void check_same_device_and_dtype(const char* op, const Tensor& a, const Tensor& b) {
  check_defined(op, a);
  check_defined(op, b);
  FG_CHECK(a.device() == b.device(), op, ": expected tensors on the same device, got ", describe(a),
           " and ", describe(b));
  FG_CHECK(a.dtype() == b.dtype(), op, ": expected tensors with the same dtype, got ", describe(a),
           " and ", describe(b));
}

inline void check_numeric(const char* op, const Tensor& t) {
  FG_CHECK(t.dtype() != DType::Bool, op, ": not supported for bool tensors, got ", describe(t));
}

inline void check_floating(const char* op, const Tensor& t) {
  FG_CHECK(is_floating(t.dtype()), op, ": expected a floating-point tensor, got ", describe(t));
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

// Runs the backend implementation for `t`'s device. CUDA kernels arrive in Phase 2.
template <typename CpuFn>
void run_on_device(const char* op, const Tensor& t, CpuFn&& cpu_fn) {
  switch (t.device().type()) {
    case DeviceType::CPU:
      cpu_fn();
      return;
    case DeviceType::CUDA:
      break;
  }
  FG_THROW(op, ": not implemented for CUDA tensors yet (Phase 2), got ", describe(t));
}

}  // namespace flashgrad::detail
