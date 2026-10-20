#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "flashgrad/core/device.h"
#include "flashgrad/core/dtype.h"
#include "flashgrad/core/shape.h"
#include "flashgrad/core/storage.h"
#include "flashgrad/core/tensor_impl.h"

namespace flashgrad {

namespace autograd {
class Node;
}  // namespace autograd

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
  float* data_ptr() const;

  // ---- Host access (synchronizes for CUDA tensors) -----------------------------------
  // The single element of a tensor with numel() == 1.
  float item() const;
  // All elements in row-major order.
  std::vector<float> to_vector() const;

  // ---- Views (share storage, never copy) ---------------------------------------------
  // Not differentiable: throws for a tensor that requires grad (with grad mode on).
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
  Tensor cpu() const { return to(Device::cpu()); }
  Tensor cuda(int index = 0) const { return to(Device::cuda(index)); }

  // ---- Autograd ----------------------------------------------------------------------
  // A leaf is a tensor that wasn't computed by a recorded op: created by the user, or
  // computed with grad mode off. Leaves that require grad get their gradient in grad()
  // after backward; outputs of recorded ops always require grad and have a grad_fn.
  bool requires_grad() const;
  // Sets requires_grad on a leaf; returns *this, e.g. randn(...).requires_grad_().
  const Tensor& requires_grad_(bool requires_grad = true) const;
  bool is_leaf() const;
  // The node that computed this tensor; null for leaves.
  std::shared_ptr<autograd::Node> grad_fn() const;
  // The gradient accumulated by backward (leaves only); undefined before the first backward.
  Tensor grad() const;
  // Replaces the gradient; an undefined tensor clears it.
  void set_grad(const Tensor& grad) const;
  // The same data without autograd history: a leaf that doesn't require grad.
  Tensor detach() const;
  // Computes the gradients of this tensor with respect to the leaves that require grad and
  // adds them to their grad(). `gradient` is the gradient of this tensor; it may be omitted
  // for a tensor with one element (it is then 1). The graph's saved tensors are freed
  // afterwards unless `retain_graph` is set. Gradients are first order only: backward
  // itself is never recorded.
  void backward(const Tensor& gradient = Tensor(), bool retain_graph = false) const;

  // ---- In-place ----------------------------------------------------------------------
  // Copies `src` (broadcast to shape()) into this tensor, moving data between devices as
  // needed.
  const Tensor& copy_(const Tensor& src) const;
  const Tensor& fill_(double value) const;  // value is rounded to float
  const Tensor& zero_() const { return fill_(0.0); }
  // Elementwise in-place arithmetic on CUDA tensors; `other` must broadcast to shape().
  // Computed into a new tensor, then copied back.
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

// Prints the values, e.g. "tensor([[1, 2], [3, 4]], dtype=float32, device=cpu)".
// Large tensors are summarized. CUDA tensors are copied to the host first.
std::ostream& operator<<(std::ostream& os, const Tensor& t);

// ---- Factories ------------------------------------------------------------------------
// The default options are float32 on the CPU, e.g. zeros({2, 3}, {.device = Device::cuda()}).

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

// Builds a tensor from host values in row-major order.
Tensor tensor(const std::vector<float>& values, Shape shape, Device device = Device::cpu());
Tensor tensor(const std::vector<float>& values, Device device = Device::cpu());  // 1-D
Tensor tensor(std::initializer_list<float> values, Shape shape, Device device = Device::cpu());
Tensor tensor(std::initializer_list<float> values, Device device = Device::cpu());  // 1-D

}  // namespace flashgrad
