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

}  // namespace
