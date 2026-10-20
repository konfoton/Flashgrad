#include "flashgrad/core/tensor.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

#include "core/checks.h"
#include "cpu/ops.h"
#include "cuda/memory.h"
#include "cuda/ops.h"

namespace flashgrad {

namespace {

// Strides that let a tensor with (sizes, strides) be viewed with `new_shape` without
// copying, or nullopt if no such strides exist. `new_shape` must have the same numel.
//
// Size-1 dims never move the offset, so they are ignored. The remaining old dims are grouped
// into blocks of neighbours laid out contiguously relative to each other
// (strides[d] == sizes[d + 1] * strides[d + 1]); a block addresses its elements like a single
// dim with `numel` elements and stride `inner_stride`. A view exists exactly when `new_shape`
// can be cut, left to right, into runs of dims whose element counts match the blocks in order.
// Each run is then laid out contiguously inside its block.
std::optional<Strides> compute_view_strides(const Shape& sizes, const Strides& strides,
                                            const Shape& new_shape) {
  if (numel(sizes) == 0) {
    return sizes == new_shape ? strides : contiguous_strides(new_shape);
  }

  struct Block {
    std::int64_t numel;
    std::int64_t inner_stride;
  };
  std::vector<Block> blocks;
  for (std::size_t d = 0; d < sizes.size(); ++d) {
    if (sizes[d] == 1) continue;
    if (!blocks.empty() && blocks.back().inner_stride == sizes[d] * strides[d]) {
      blocks.back().numel *= sizes[d];
      blocks.back().inner_stride = strides[d];
    } else {
      blocks.push_back({sizes[d], strides[d]});
    }
  }

  Strides new_strides(new_shape.size());
  std::size_t d = 0;  // next new dim to assign
  for (const Block& block : blocks) {
    const std::size_t first = d;
    std::int64_t count = 1;
    while (count < block.numel && d < new_shape.size()) count *= new_shape[d++];
    if (count != block.numel) return std::nullopt;
    std::int64_t stride = block.inner_stride;
    for (std::size_t k = d; k-- > first;) {
      new_strides[k] = stride;
      stride *= new_shape[k];
    }
  }
  // Only size-1 dims can be left over; any stride works for them.
  for (; d < new_shape.size(); ++d) {
    if (new_shape[d] != 1) return std::nullopt;
    new_strides[d] = 1;
  }
  return new_strides;
}

// Replaces a single -1 in `shape` so that it holds `total` elements.
Shape infer_shape(Shape shape, std::int64_t total, const char* op) {
  std::optional<std::size_t> inferred;
  std::int64_t known = 1;
  for (std::size_t d = 0; d < shape.size(); ++d) {
    if (shape[d] == -1) {
      FG_CHECK(!inferred, op, ": only one dimension can be -1, got shape ", to_string(shape));
      inferred = d;
    } else {
      FG_CHECK(shape[d] >= 0, op, ": invalid shape ", to_string(shape));
      known *= shape[d];
    }
  }
  if (inferred) {
    FG_CHECK(known != 0 && total % known == 0, op, ": shape ", to_string(shape),
             " is invalid for a tensor with ", total, " elements");
    shape[*inferred] = total / known;
  } else {
    FG_CHECK(known == total, op, ": shape ", to_string(shape), " is invalid for a tensor with ",
             total, " elements");
  }
  return shape;
}

// Copies a CUDA tensor of any layout to the host: downloads the contiguous storage range
// that the view spans and rebuilds the same view on top of it.
Tensor download(const Tensor& src) {
  if (src.numel() == 0) return empty(src.shape(), {.dtype = src.dtype(), .device = Device::cpu()});
  std::int64_t span = 1;
  for (std::int64_t d = 0; d < src.dim(); ++d) span += (src.size(d) - 1) * src.stride(d);
  const std::size_t nbytes = static_cast<std::size_t>(span) * src.element_size();
  Storage host(nbytes, Device::cpu());
  cuda::detail::copy_device_to_host(host.data(), src.data_ptr(), src.device().index(), nbytes);
  return Tensor(
      std::make_shared<TensorImpl>(std::move(host), src.shape(), src.strides(), 0, src.dtype()));
}

}  // namespace

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

const TensorImpl& Tensor::impl_ref() const {
  FG_CHECK(impl_ != nullptr, "Tensor: operation on an undefined tensor");
  return *impl_;
}

// ---- Metadata ------------------------------------------------------------------------

const Shape& Tensor::shape() const { return impl_ref().shape(); }
const Strides& Tensor::strides() const { return impl_ref().strides(); }
std::int64_t Tensor::dim() const { return static_cast<std::int64_t>(shape().size()); }

std::int64_t Tensor::size(std::int64_t d) const { return shape()[normalize_dim(d, dim(), "size")]; }

std::int64_t Tensor::stride(std::int64_t d) const {
  return strides()[normalize_dim(d, dim(), "stride")];
}

std::int64_t Tensor::numel() const { return impl_ref().numel(); }
std::int64_t Tensor::offset() const { return impl_ref().offset(); }
DType Tensor::dtype() const { return impl_ref().dtype(); }
Device Tensor::device() const { return impl_ref().device(); }
TensorOptions Tensor::options() const { return {.dtype = dtype(), .device = device()}; }
const Storage& Tensor::storage() const { return impl_ref().storage(); }
std::size_t Tensor::element_size() const { return flashgrad::element_size(dtype()); }
std::size_t Tensor::nbytes() const { return static_cast<std::size_t>(numel()) * element_size(); }

bool Tensor::is_contiguous() const {
  if (numel() == 0) return true;
  std::int64_t expected = 1;
  for (std::int64_t d = dim() - 1; d >= 0; --d) {
    const std::int64_t s = shape()[d];
    if (s == 1) continue;  // the stride of a size-1 dim never matters
    if (strides()[d] != expected) return false;
    expected *= s;
  }
  return true;
}

float* Tensor::data_ptr() const {
  auto* base = static_cast<float*>(storage().data());
  if (base == nullptr) return nullptr;
  return base + offset();
}

// ---- Host access -----------------------------------------------------------------------

float Tensor::item() const {
  FG_CHECK(numel() == 1, "item: expected a tensor with one element, got ", describe(*this));
  return *cpu().data_ptr();
}

std::vector<float> Tensor::to_vector() const {
  const Tensor host = cpu().contiguous();
  const float* data = host.data_ptr();
  if (data == nullptr) return {};
  return std::vector<float>(data, data + host.numel());
}

// ---- Views -----------------------------------------------------------------------------

Tensor Tensor::as_strided(Shape new_shape, Strides new_strides, std::int64_t new_offset) const {
  return Tensor(std::make_shared<TensorImpl>(storage(), std::move(new_shape),
                                             std::move(new_strides), new_offset, dtype()));
}

Tensor Tensor::view(Shape new_shape) const {
  new_shape = infer_shape(std::move(new_shape), numel(), "view");
  const auto new_strides = compute_view_strides(shape(), strides(), new_shape);
  FG_CHECK(new_strides.has_value(), "view: cannot view ", describe(*this), " with strides ",
           to_string(strides()), " as shape ", to_string(new_shape),
           " without copying; use reshape()");
  return as_strided(std::move(new_shape), *new_strides, offset());
}

Tensor Tensor::reshape(Shape new_shape) const {
  new_shape = infer_shape(std::move(new_shape), numel(), "reshape");
  if (compute_view_strides(shape(), strides(), new_shape)) return view(std::move(new_shape));
  return clone().view(std::move(new_shape));
}

Tensor Tensor::flatten(std::int64_t start_dim, std::int64_t end_dim) const {
  if (dim() == 0) return reshape({1});
  const std::int64_t start = normalize_dim(start_dim, dim(), "flatten");
  const std::int64_t end = normalize_dim(end_dim, dim(), "flatten");
  FG_CHECK(start <= end, "flatten: start_dim ", start_dim, " comes after end_dim ", end_dim,
           " for ", describe(*this));
  Shape new_shape(shape().begin(), shape().begin() + start);
  std::int64_t flat = 1;
  for (std::int64_t d = start; d <= end; ++d) flat *= shape()[d];
  new_shape.push_back(flat);
  new_shape.insert(new_shape.end(), shape().begin() + end + 1, shape().end());
  return reshape(std::move(new_shape));
}

Tensor Tensor::permute(const std::vector<std::int64_t>& dims) const {
  FG_CHECK(static_cast<std::int64_t>(dims.size()) == dim(), "permute: got ", dims.size(), " dims ",
           to_string(dims), " for ", describe(*this));
  Shape new_shape(dims.size());
  Strides new_strides(dims.size());
  std::vector<bool> seen(dims.size(), false);
  for (std::size_t i = 0; i < dims.size(); ++i) {
    const std::int64_t d = normalize_dim(dims[i], dim(), "permute");
    FG_CHECK(!seen[d], "permute: dimension ", d, " repeated in ", to_string(dims));
    seen[d] = true;
    new_shape[i] = shape()[d];
    new_strides[i] = strides()[d];
  }
  return as_strided(std::move(new_shape), std::move(new_strides), offset());
}

Tensor Tensor::transpose(std::int64_t dim0, std::int64_t dim1) const {
  const std::int64_t d0 = normalize_dim(dim0, dim(), "transpose");
  const std::int64_t d1 = normalize_dim(dim1, dim(), "transpose");
  Shape new_shape = shape();
  Strides new_strides = strides();
  std::swap(new_shape[d0], new_shape[d1]);
  std::swap(new_strides[d0], new_strides[d1]);
  return as_strided(std::move(new_shape), std::move(new_strides), offset());
}

Tensor Tensor::t() const {
  FG_CHECK(dim() <= 2, "t: expected a tensor with at most 2 dimensions, got ", describe(*this));
  return dim() < 2 ? *this : transpose(0, 1);
}

Tensor Tensor::unsqueeze(std::int64_t d) const {
  const std::int64_t pos = normalize_dim(d, dim() + 1, "unsqueeze");
  Shape new_shape = shape();
  Strides new_strides = strides();
  const std::int64_t new_stride = pos < dim() ? shape()[pos] * strides()[pos] : 1;
  new_shape.insert(new_shape.begin() + pos, 1);
  new_strides.insert(new_strides.begin() + pos, new_stride);
  return as_strided(std::move(new_shape), std::move(new_strides), offset());
}

Tensor Tensor::squeeze() const {
  Shape new_shape;
  Strides new_strides;
  for (std::int64_t d = 0; d < dim(); ++d) {
    if (shape()[d] == 1) continue;
    new_shape.push_back(shape()[d]);
    new_strides.push_back(strides()[d]);
  }
  return as_strided(std::move(new_shape), std::move(new_strides), offset());
}

Tensor Tensor::squeeze(std::int64_t d) const {
  const std::int64_t pos = normalize_dim(d, dim(), "squeeze");
  if (shape()[pos] != 1) return *this;
  Shape new_shape = shape();
  Strides new_strides = strides();
  new_shape.erase(new_shape.begin() + pos);
  new_strides.erase(new_strides.begin() + pos);
  return as_strided(std::move(new_shape), std::move(new_strides), offset());
}

Tensor Tensor::expand(const Shape& target) const {
  const auto ndim = static_cast<std::int64_t>(target.size());
  FG_CHECK(ndim >= dim(), "expand: target shape ", to_string(target), " has fewer dimensions than ",
           describe(*this));
  const std::int64_t lead = ndim - dim();
  Shape new_shape(target.size());
  Strides new_strides(target.size());
  for (std::int64_t d = 0; d < ndim; ++d) {
    if (d < lead) {
      FG_CHECK(target[d] >= 0, "expand: invalid size ", target[d], " for new dimension ", d, " in ",
               to_string(target));
      new_shape[d] = target[d];
      new_strides[d] = 0;
      continue;
    }
    const std::int64_t current = shape()[d - lead];
    const std::int64_t want = target[d] == -1 ? current : target[d];
    FG_CHECK(want >= 0, "expand: invalid size ", target[d], " in ", to_string(target));
    FG_CHECK(want == current || current == 1, "expand: cannot expand ", describe(*this), " to ",
             to_string(target), " (dimension ", d, ")");
    new_shape[d] = want;
    new_strides[d] = want == current ? strides()[d - lead] : 0;
  }
  return as_strided(std::move(new_shape), std::move(new_strides), offset());
}

Tensor Tensor::slice(std::int64_t d, std::int64_t start, std::int64_t end,
                     std::int64_t step) const {
  const std::int64_t pos = normalize_dim(d, dim(), "slice");
  FG_CHECK(step > 0, "slice: step must be positive, got ", step);
  const std::int64_t n = shape()[pos];
  if (start < 0) start += n;
  if (end < 0) end += n;
  start = std::clamp<std::int64_t>(start, 0, n);
  end = std::clamp<std::int64_t>(end, start, n);
  Shape new_shape = shape();
  Strides new_strides = strides();
  new_shape[pos] = (end - start + step - 1) / step;
  new_strides[pos] = strides()[pos] * step;
  return as_strided(std::move(new_shape), std::move(new_strides),
                    offset() + start * strides()[pos]);
}

Tensor Tensor::select(std::int64_t d, std::int64_t index) const {
  const std::int64_t pos = normalize_dim(d, dim(), "select");
  const std::int64_t n = shape()[pos];
  FG_CHECK(index >= -n && index < n, "select: index ", index, " out of range for dimension ", d,
           " of ", describe(*this));
  if (index < 0) index += n;
  Shape new_shape = shape();
  Strides new_strides = strides();
  new_shape.erase(new_shape.begin() + pos);
  new_strides.erase(new_strides.begin() + pos);
  return as_strided(std::move(new_shape), std::move(new_strides),
                    offset() + index * strides()[pos]);
}

// ---- Copies and conversions --------------------------------------------------------------

Tensor Tensor::contiguous() const { return is_contiguous() ? *this : clone(); }

Tensor Tensor::clone() const {
  Tensor out = empty(shape(), options());
  out.copy_(*this);
  return out;
}

Tensor Tensor::to(Device target) const {
  if (device() == target) return *this;
  Tensor out = empty(shape(), {.dtype = dtype(), .device = target});
  out.copy_(*this);
  return out;
}

const Tensor& Tensor::copy_(const Tensor& src) const {
  detail::check_defined("copy_", *this);
  detail::check_defined("copy_", src);
  FG_CHECK(broadcast_shapes(src.shape(), shape(), "copy_") == shape(), "copy_: cannot broadcast ",
           describe(src), " to ", describe(*this));
  if (numel() == 0) return *this;
  detail::check_writable("copy_", *this);
  // Reading from memory that is being written would make the result order-dependent.
  const Tensor source = src.storage().is_same(storage()) ? src.clone() : src;

  if (device().is_cpu() && source.device().is_cpu()) {
    cpu::copy(*this, source.expand(shape()));
  } else if (device().is_cpu()) {  // CUDA -> CPU
    cpu::copy(*this, download(source).expand(shape()));
  } else if (source.device().is_cpu()) {  // CPU -> CUDA
    FG_CHECK(is_contiguous(), "copy_: copying into a non-contiguous CUDA tensor needs the CUDA ",
             "kernels of Phase 2, got destination ", describe(*this));
    // Lay out the data contiguously on the host, then upload it in one transfer.
    const Tensor staging = empty(shape(), {.dtype = dtype(), .device = Device::cpu()});
    cpu::copy(staging, source.expand(shape()));
    cuda::detail::copy_host_to_device(data_ptr(), device().index(), staging.data_ptr(), nbytes());
  } else {  // CUDA -> CUDA
    FG_CHECK(shape() == source.shape() && is_contiguous() && source.is_contiguous(),
             "copy_: CUDA-to-CUDA copies with broadcasting or non-contiguous ",
             "tensors need the CUDA kernels of Phase 2, got ", describe(source), " -> ",
             describe(*this));
    cuda::detail::copy_device_to_device(data_ptr(), device().index(), source.data_ptr(),
                                        source.device().index(), nbytes());
  }
  return *this;
}

const Tensor& Tensor::fill_(double value) const {
  detail::check_defined("fill_", *this);
  if (numel() == 0) return *this;
  if (device().is_cpu()) {
    cpu::fill(*this, value);
  } else {
    FG_CHECK(is_contiguous() || dim() <= cuda::detail::kMaxDims, "fill_: at most ",
             cuda::detail::kMaxDims, " dimensions are supported for non-contiguous CUDA ",
             "tensors, got ", describe(*this), " with strides ", to_string(strides()));
    cuda::detail::fill(*this, static_cast<float>(value));
  }
  return *this;
}

std::string describe(const Tensor& t) {
  if (!t.defined()) return "Tensor(undefined)";
  std::ostringstream oss;
  oss << "Tensor(shape=" << to_string(t.shape()) << ", dtype=" << t.dtype()
      << ", device=" << t.device() << ")";
  return oss.str();
}

}  // namespace flashgrad
