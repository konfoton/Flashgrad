#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <vector>

#include "flashgrad/core/ops.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "gradcheck.h"
#include "reference.h"

// Gradients of the view and copy ops. The references pick the view's elements out of the
// row-major input with explicitly computed strides.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
using fg::Shape;
using fg::Tensor;
using gradcheck::Values;

using UnaryOp = std::function<Tensor(const Tensor&)>;

// Checks op(a) against `reference`, which maps the row-major values of `a` to those of the
// result.
void check(const UnaryOp& op, const gradcheck::Reference& reference, const Tensor& a) {
  const auto wrapped = [&](const std::vector<Tensor>& in) { return op(in[0]); };
  EXPECT_TRUE(gradcheck::check(wrapped, reference, {a})) << fg::describe(a);
}

Values same_values(const std::vector<Values>& in) { return in[0]; }

// ---- Copies ---------------------------------------------------------------------------

TEST(ViewGrads, CloneAndContiguous) {
  SKIP_IF_NO_CUDA();
  const Tensor a = ref::random_cuda({3, 4}, 1);
  check([](const Tensor& t) { return t.clone(); }, same_values, a);
  check([](const Tensor& t) { return t.contiguous(); }, same_values, a);  // returns t itself
  check([](const Tensor& t) { return t.contiguous(); }, same_values,
        ref::random_cuda({4, 3}, 2).t());  // copies
}

TEST(ViewGrads, ToMovesTheGradientBack) {
  SKIP_IF_NO_CUDA();
  // A CPU leaf moved to the GPU gets its gradient on the CPU.
  const Tensor host = fg::tensor({1, 2, 3}).requires_grad_();
  const Tensor dev = host.cuda();
  EXPECT_TRUE(dev.requires_grad());
  (dev * dev).backward(fg::ones({3}, {.device = fg::Device::cuda()}));
  EXPECT_TRUE(host.grad().device().is_cpu());
  EXPECT_EQ(host.grad().to_vector(), (std::vector<float>{2, 4, 6}));
  // And the other way round.
  const Tensor gpu_leaf = ref::random_cuda({2, 2}, 3);
  check([](const Tensor& t) { return t.cpu().cuda(); }, same_values, gpu_leaf);
}

// ---- Reshaping ------------------------------------------------------------------------

TEST(ViewGrads, ViewReshapeFlatten) {
  SKIP_IF_NO_CUDA();
  const Tensor a = ref::random_cuda({2, 3, 4}, 10);
  // Row-major order is unchanged, so every reference is the identity.
  check([](const Tensor& t) { return t.view({6, 4}); }, same_values, a);
  check([](const Tensor& t) { return t.view({-1}); }, same_values, a);
  check([](const Tensor& t) { return t.reshape({4, 3, 2}); }, same_values, a);
  check([](const Tensor& t) { return t.flatten(1); }, same_values, a);
  check([](const Tensor& t) { return t.flatten(); }, same_values, ref::random_cuda({}, 11));
  // A non-contiguous input: reshape copies, then views the copy.
  check([](const Tensor& t) { return t.reshape({-1}); }, same_values,
        ref::random_cuda({4, 5}, 12).t());
}

// ---- Permutations ---------------------------------------------------------------------

// The row-major strides of `shape`, reordered by `dims`.
std::vector<std::int64_t> permuted_strides(const Shape& shape,
                                           const std::vector<std::int64_t>& dims) {
  const std::vector<std::int64_t> strides = ref::row_major_strides(shape);
  std::vector<std::int64_t> out;
  for (const std::int64_t d : dims) out.push_back(strides[static_cast<std::size_t>(d)]);
  return out;
}

TEST(ViewGrads, PermuteTransposeT) {
  SKIP_IF_NO_CUDA();
  const Shape shape{2, 3, 4};
  const Tensor a = ref::random_cuda(shape, 20);
  const auto permuted = [&](std::vector<std::int64_t> dims) {
    return [shape, dims](const std::vector<Values>& in) {
      Shape out_shape;
      for (const std::int64_t d : dims) out_shape.push_back(shape[static_cast<std::size_t>(d)]);
      return ref::gather(in[0], out_shape, permuted_strides(shape, dims));
    };
  };
  check([](const Tensor& t) { return t.permute({2, 0, 1}); }, permuted({2, 0, 1}), a);
  check([](const Tensor& t) { return t.permute({1, 2, 0}); }, permuted({1, 2, 0}), a);
  check([](const Tensor& t) { return t.transpose(0, 2); }, permuted({2, 1, 0}), a);
  check([](const Tensor& t) { return t.transpose(-1, 1); }, permuted({0, 2, 1}), a);

  const Shape matrix{3, 5};
  check([](const Tensor& t) { return t.t(); },
        [&](const std::vector<Values>& in) {
          return ref::gather(in[0], {5, 3}, permuted_strides(matrix, {1, 0}));
        },
        ref::random_cuda(matrix, 21));
}

// ---- Size-1 dimensions ----------------------------------------------------------------

TEST(ViewGrads, SqueezeUnsqueeze) {
  SKIP_IF_NO_CUDA();
  // Adding or removing size-1 dims keeps the row-major order.
  const Tensor a = ref::random_cuda({2, 1, 3, 1}, 30);
  check([](const Tensor& t) { return t.squeeze(); }, same_values, a);
  check([](const Tensor& t) { return t.squeeze(1); }, same_values, a);
  check([](const Tensor& t) { return t.squeeze(-1); }, same_values, a);
  check([](const Tensor& t) { return t.squeeze(0); }, same_values, a);  // size 2: unchanged
  check([](const Tensor& t) { return t.unsqueeze(0); }, same_values, a);
  check([](const Tensor& t) { return t.unsqueeze(4); }, same_values, a);
  check([](const Tensor& t) { return t.unsqueeze(-2).squeeze(); }, same_values,
        ref::random_cuda({5, 2}, 31).t());
}

// ---- Broadcasting ---------------------------------------------------------------------

TEST(ViewGrads, Expand) {
  SKIP_IF_NO_CUDA();
  const auto expanded = [](Shape from, Shape to) {
    return [from, to](const std::vector<Values>& in) {
      // Stride 0 for new and stretched dims, the row-major stride otherwise.
      const std::vector<std::int64_t> from_strides = ref::row_major_strides(from);
      const std::size_t lead = to.size() - from.size();
      std::vector<std::int64_t> strides(to.size(), 0);
      for (std::size_t d = lead; d < to.size(); ++d) {
        if (from[d - lead] == to[d]) strides[d] = from_strides[d - lead];
      }
      return ref::gather(in[0], to, strides);
    };
  };
  check([](const Tensor& t) { return t.expand({4, 3}); }, expanded({1, 3}, {4, 3}),
        ref::random_cuda({1, 3}, 40));
  check([](const Tensor& t) { return t.expand({2, 3, 4}); }, expanded({3, 1}, {2, 3, 4}),
        ref::random_cuda({3, 1}, 41));
  check([](const Tensor& t) { return t.expand({5}); }, expanded({}, {5}), ref::random_cuda({}, 42));
  check([](const Tensor& t) { return t.expand({2, -1}); }, expanded({3}, {2, 3}),
        ref::random_cuda({3}, 43));
}

}  // namespace
