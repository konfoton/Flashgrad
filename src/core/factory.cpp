#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "flashgrad/core/tensor.h"

namespace flashgrad {

Tensor empty(Shape shape, TensorOptions options) {
  const std::int64_t n = numel(shape);
  Storage storage(static_cast<std::size_t>(n) * element_size(options.dtype), options.device);
  Strides strides = contiguous_strides(shape);
  return Tensor(std::make_shared<TensorImpl>(std::move(storage), std::move(shape),
                                             std::move(strides), 0, options.dtype));
}

Tensor zeros(Shape shape, TensorOptions options) { return full(std::move(shape), 0.0, options); }

Tensor ones(Shape shape, TensorOptions options) { return full(std::move(shape), 1.0, options); }

Tensor full(Shape shape, double value, TensorOptions options) {
  Tensor out = empty(std::move(shape), options);
  out.fill_(value);
  return out;
}

Tensor arange(double start, double end, double step, TensorOptions options) {
  FG_CHECK(step != 0.0, "arange: step must be non-zero");
  FG_CHECK(std::isfinite(start) && std::isfinite(end) && std::isfinite(step),
           "arange: start, end and step must be finite, got ", start, ", ", end, ", ", step);
  const double count = std::ceil((end - start) / step);
  const std::int64_t n = count > 0.0 ? static_cast<std::int64_t>(count) : 0;
  Tensor out = empty({n}, {.dtype = options.dtype, .device = Device::cpu()});
  float* data = out.data_ptr();
  for (std::int64_t i = 0; i < n; ++i) {
    data[i] = static_cast<float>(start + static_cast<double>(i) * step);
  }
  return out.to(options.device);
}

Tensor arange(double end, TensorOptions options) { return arange(0.0, end, 1.0, options); }

Tensor eye(std::int64_t n, TensorOptions options) {
  FG_CHECK(n >= 0, "eye: size must be non-negative, got ", n);
  Tensor out = zeros({n, n}, {.dtype = options.dtype, .device = Device::cpu()});
  float* data = out.data_ptr();
  for (std::int64_t i = 0; i < n; ++i) data[i * n + i] = 1.0f;
  return out.to(options.device);
}

Tensor empty_like(const Tensor& t) { return empty(t.shape(), t.options()); }
Tensor zeros_like(const Tensor& t) { return zeros(t.shape(), t.options()); }
Tensor ones_like(const Tensor& t) { return ones(t.shape(), t.options()); }
Tensor full_like(const Tensor& t, double value) { return full(t.shape(), value, t.options()); }

Tensor tensor(const std::vector<float>& values, Shape shape, Device device) {
  FG_CHECK(numel(shape) == static_cast<std::int64_t>(values.size()), "tensor: ", values.size(),
           " values do not fill shape ", to_string(shape));
  Tensor out = empty(std::move(shape), {.device = Device::cpu()});
  std::copy(values.begin(), values.end(), out.data_ptr());
  return out.to(device);
}

Tensor tensor(const std::vector<float>& values, Device device) {
  return tensor(values, Shape{static_cast<std::int64_t>(values.size())}, device);
}

Tensor tensor(std::initializer_list<float> values, Shape shape, Device device) {
  return tensor(std::vector<float>(values), std::move(shape), device);
}

Tensor tensor(std::initializer_list<float> values, Device device) {
  return tensor(std::vector<float>(values), device);
}

}  // namespace flashgrad
