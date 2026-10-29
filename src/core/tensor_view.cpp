#include "flashgrad/core/tensor_view.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "autograd/grad_state.h"
#include "flashgrad/core/error.h"

namespace flashgrad {

TensorView::TensorView(Storage storage, Shape shape, Strides strides, std::int64_t offset,
                       DType dtype)
    : storage_(std::move(storage)),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset),
      dtype_(dtype),
      numel_(flashgrad::numel(shape_)) {
  FG_CHECK(storage_.defined(), "TensorView: storage is undefined");
  FG_CHECK(shape_.size() == strides_.size(), "TensorView: shape ", to_string(shape_),
           " and strides ", to_string(strides_), " have different lengths");
  FG_CHECK(offset_ >= 0, "TensorView: negative storage offset ", offset_);
  for (const std::int64_t stride : strides_) {
    FG_CHECK(stride >= 0, "TensorView: negative strides are not supported, got ",
             to_string(strides_));
  }
  if (numel_ > 0) {
    std::int64_t last = offset_;
    for (std::size_t d = 0; d < shape_.size(); ++d) last += (shape_[d] - 1) * strides_[d];
    const auto capacity = static_cast<std::int64_t>(storage_.nbytes() / element_size(dtype_));
    FG_CHECK(last < capacity, "TensorView: view with shape ", to_string(shape_), ", strides ",
             to_string(strides_), " and offset ", offset_, " needs ", last + 1,
             " elements, but the storage holds only ", capacity, " ", dtype_, " elements");
  }
}

TensorView::~TensorView() = default;

autograd::GradState& TensorView::ensure_grad_state() {
  if (!grad_state_) grad_state_ = std::make_unique<autograd::GradState>();
  return *grad_state_;
}

}  // namespace flashgrad
