#pragma once

#include <cstdint>
#include <memory>

#include "flashgrad/core/device.h"
#include "flashgrad/core/dtype.h"
#include "flashgrad/core/shape.h"
#include "flashgrad/core/storage.h"

namespace flashgrad {

namespace autograd {
struct GradState;
}  // namespace autograd

// The data behind a Tensor handle: a strided view into a Storage.
// Element (i0, i1, ...) lives at element index offset + i0 * strides[0] + i1 * strides[1] + ...
// of the storage. The constructor validates that every element of the view is in bounds.
// Tensors that take part in autograd also carry a GradState (src/autograd/grad_state.h).
class TensorView {
 public:
  TensorView(Storage storage, Shape shape, Strides strides, std::int64_t offset, DType dtype);
  ~TensorView();
  TensorView(const TensorView&) = delete;
  TensorView& operator=(const TensorView&) = delete;

  const Storage& storage() const noexcept { return storage_; }
  const Shape& shape() const noexcept { return shape_; }
  const Strides& strides() const noexcept { return strides_; }
  std::int64_t offset() const noexcept { return offset_; }
  DType dtype() const noexcept { return dtype_; }
  Device device() const noexcept { return storage_.device(); }
  std::int64_t numel() const noexcept { return numel_; }

  // Null until the tensor takes part in autograd.
  autograd::GradState* grad_state() const noexcept { return grad_state_.get(); }
  autograd::GradState& ensure_grad_state();

 private:
  Storage storage_;
  Shape shape_;
  Strides strides_;
  std::int64_t offset_;
  DType dtype_;
  std::int64_t numel_;
  std::unique_ptr<autograd::GradState> grad_state_;
};

}  // namespace flashgrad
