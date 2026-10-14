#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "flashgrad/core/device.h"
#include "flashgrad/core/dtype.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/shape.h"
#include "flashgrad/core/storage.h"
#include "flashgrad/core/tensor_impl.h"

namespace flashgrad {

struct TensorOptions {
  DType dtype = DType::Float32;
  Device device = Device::cpu();
};

// A handle to a TensorImpl. Copying a Tensor is cheap and shares the data (like PyTorch);
// use clone() for a deep copy. View operations (view, permute, slice, ...) return new
// handles that share the storage of the original tensor.
//
// Methods are const because they don't change which tensor the handle refers to; in-place
// methods (suffix `_`) modify the shared data.
class Tensor {
 public:
  Tensor() = default;  // undefined tensor
  explicit Tensor(std::shared_ptr<TensorImpl> impl);

  bool defined() const noexcept { return impl_ != nullptr; }
  const std::shared_ptr<TensorImpl>& impl() const noexcept { return impl_; }
  // True when both handles refer to the same TensorImpl.
  bool is_same(const Tensor& other) const noexcept { return impl_ == other.impl_; }

  // ---- Metadata ----------------------------------------------------------------------
  const Shape& shape() const;
  const Strides& strides() const;
  std::int64_t dim() const;
  std::int64_t size(std::int64_t dim) const;    // negative dims count from the end
  std::int64_t stride(std::int64_t dim) const;  // negative dims count from the end
  std::int64_t numel() const;
  std::int64_t offset() const;  // in elements, into storage()
  DType dtype() const;
  Device device() const;
  TensorOptions options() const;
  const Storage& storage() const;
  std::size_t element_size() const;
  std::size_t nbytes() const;  // numel() * element_size()
  bool is_contiguous() const;

  // ---- Raw data ----------------------------------------------------------------------
  // Pointer to the first element (offset applied). A device pointer for CUDA tensors;
  // nullptr for tensors without elements.
  void* data_ptr() const;
  template <typename T>
  T* data_ptr() const;

  // ---- Host access (synchronizes for CUDA tensors) -----------------------------------
  // The single element of a tensor with numel() == 1, converted to T.
  template <typename T>
  T item() const;
  // All elements in row-major order. T must match dtype() exactly.
  template <typename T>
  std::vector<T> to_vector() const;

  // ---- Views (share storage, never copy) ---------------------------------------------
  Tensor as_strided(Shape shape, Strides strides, std::int64_t offset) const;
  // One dimension may be -1 and is inferred. Throws if the strides don't allow a view.
  Tensor view(Shape shape) const;
  Tensor permute(const std::vector<std::int64_t>& dims) const;
  Tensor transpose(std::int64_t dim0, std::int64_t dim1) const;
  Tensor t() const;  // transpose of a tensor with at most 2 dimensions
  Tensor unsqueeze(std::int64_t dim) const;
  Tensor squeeze() const;  // removes all size-1 dimensions
  Tensor squeeze(std::int64_t dim) const;
  // Broadcasts size-1 dims (stride 0) and prepends new leading dims; -1 keeps a size.
  Tensor expand(const Shape& shape) const;
  // Python-style slice [start:end:step] along `dim`; negative indices count from the end,
  // out-of-range bounds are clamped. `step` must be positive.
  Tensor slice(std::int64_t dim, std::int64_t start,
               std::int64_t end = std::numeric_limits<std::int64_t>::max(),
               std::int64_t step = 1) const;
  Tensor narrow(std::int64_t dim, std::int64_t start, std::int64_t length) const;
  // Removes `dim` by picking one index along it.
  Tensor select(std::int64_t dim, std::int64_t index) const;
  Tensor operator[](std::int64_t index) const { return select(0, index); }

  // ---- Views when possible, otherwise copies -----------------------------------------
  Tensor reshape(Shape shape) const;
  Tensor flatten(std::int64_t start_dim = 0, std::int64_t end_dim = -1) const;
  Tensor contiguous() const;  // returns *this when already contiguous

  // ---- Copies and conversions --------------------------------------------------------
  Tensor clone() const;            // contiguous deep copy
  Tensor to(Device device) const;  // returns *this when already on `device`
  Tensor to(DType dtype) const;    // returns *this when already `dtype`
  Tensor cpu() const { return to(Device::cpu()); }
  Tensor cuda(int index = 0) const { return to(Device::cuda(index)); }

  // ---- In-place ----------------------------------------------------------------------
  // Copies `src` (broadcast to shape()) into this tensor, converting the dtype and moving
  // data between devices as needed.
  const Tensor& copy_(const Tensor& src) const;
  const Tensor& fill_(double value) const;  // value is converted to dtype()
  const Tensor& zero_() const { return fill_(0.0); }
  // Elementwise in-place arithmetic; `other` must broadcast to shape().
  const Tensor& add_(const Tensor& other) const;
  const Tensor& sub_(const Tensor& other) const;
  const Tensor& mul_(const Tensor& other) const;
  const Tensor& div_(const Tensor& other) const;
  const Tensor& add_(double other) const;
  const Tensor& sub_(double other) const;
  const Tensor& mul_(double other) const;
  const Tensor& div_(double other) const;

 private:
  const TensorImpl& impl_ref() const;

  std::shared_ptr<TensorImpl> impl_;
};

// "Tensor(shape=[2, 3], dtype=float32, device=cpu)" — used in error messages.
std::string describe(const Tensor& t);

// Prints the values, e.g. "tensor([[1, 2], [3, 4]], dtype=int32, device=cpu)".
// Large tensors are summarized. CUDA tensors are copied to the host first.
std::ostream& operator<<(std::ostream& os, const Tensor& t);

// ---- Factories ------------------------------------------------------------------------
// The default options are float32 on the CPU, e.g. zeros({2, 3}, {.dtype = DType::Int64}).

Tensor empty(Shape shape, TensorOptions options = {});  // uninitialized
Tensor zeros(Shape shape, TensorOptions options = {});
Tensor ones(Shape shape, TensorOptions options = {});
Tensor full(Shape shape, double value, TensorOptions options = {});
// Values start, start + step, ... up to (excluding) end.
Tensor arange(double start, double end, double step = 1.0, TensorOptions options = {});
Tensor arange(double end, TensorOptions options = {});
Tensor eye(std::int64_t n, TensorOptions options = {});  // n x n identity

Tensor empty_like(const Tensor& t);
Tensor zeros_like(const Tensor& t);
Tensor ones_like(const Tensor& t);
Tensor full_like(const Tensor& t, double value);

// Builds a tensor from host values; the dtype follows T (float -> float32, int -> int32,
// std::int64_t -> int64, double -> float64, bool -> bool).
template <typename T>
Tensor tensor(const std::vector<T>& values, Shape shape, Device device = Device::cpu());
template <typename T>
Tensor tensor(const std::vector<T>& values, Device device = Device::cpu()) {
  return tensor(values, Shape{static_cast<std::int64_t>(values.size())}, device);
}
template <typename T>
Tensor tensor(std::initializer_list<T> values, Shape shape, Device device = Device::cpu()) {
  return tensor(std::vector<T>(values), std::move(shape), device);
}
template <typename T>
Tensor tensor(std::initializer_list<T> values, Device device = Device::cpu()) {
  return tensor(std::vector<T>(values), device);
}

// ---- Template definitions -------------------------------------------------------------

template <typename T>
T* Tensor::data_ptr() const {
  FG_CHECK(dtype_of<std::remove_cv_t<T>> == dtype(), "data_ptr: requested ",
           dtype_of<std::remove_cv_t<T>>, " pointer but tensor is ", describe(*this));
  return static_cast<T*>(data_ptr());
}

template <typename T>
T Tensor::item() const {
  FG_CHECK(numel() == 1, "item: expected a tensor with one element, got ", describe(*this));
  const Tensor host = cpu();
  return dispatch_all(dtype(), "item",
                      [&]<typename S>() { return static_cast<T>(*host.data_ptr<S>()); });
}

template <typename T>
std::vector<T> Tensor::to_vector() const {
  FG_CHECK(dtype_of<T> == dtype(), "to_vector: requested ", dtype_of<T>, " values but tensor is ",
           describe(*this));
  const Tensor host = cpu().contiguous();
  const T* data = host.data_ptr<T>();
  if (data == nullptr) return {};
  return std::vector<T>(data, data + host.numel());
}

template <typename T>
Tensor tensor(const std::vector<T>& values, Shape shape, Device device) {
  FG_CHECK(numel(shape) == static_cast<std::int64_t>(values.size()), "tensor: ", values.size(),
           " values do not fill shape ", to_string(shape));
  Tensor out = empty(std::move(shape), {.dtype = dtype_of<T>, .device = Device::cpu()});
  std::copy(values.begin(), values.end(), out.data_ptr<T>());
  return out.to(device);
}

}  // namespace flashgrad
