#include <gtest/gtest.h>

#include <vector>

#include "flashgrad/autograd/grad_mode.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"

// In-place ops aren't recorded, so autograd forbids them on tensors that require grad.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
using fg::Tensor;

const fg::TensorOptions kGpu{.device = fg::Device::cuda()};

TEST(InPlaceAutograd, RejectsWritesToTensorsThatRequireGrad) {
  SKIP_IF_NO_CUDA();
  const Tensor w = fg::ones({3}, kGpu).requires_grad_();
  EXPECT_THROW(w.fill_(0.0), fg::Error);
  EXPECT_THROW(w.zero_(), fg::Error);
  EXPECT_THROW(w.copy_(fg::zeros({3}, kGpu)), fg::Error);
  EXPECT_THROW(w.add_(1.0), fg::Error);
  EXPECT_THROW(w.mul_(fg::ones({3}, kGpu)), fg::Error);
  // Outputs of recorded ops and views of w require grad too.
  EXPECT_THROW((w * 2.0f).add_(1.0), fg::Error);
  EXPECT_THROW(w.slice(0, 0, 2).fill_(1.0), fg::Error);
  EXPECT_EQ(w.to_vector(), (std::vector<float>{1, 1, 1}));  // nothing changed
}

TEST(InPlaceAutograd, RejectsSourcesThatRequireGrad) {
  SKIP_IF_NO_CUDA();
  const Tensor w = fg::ones({3}, kGpu).requires_grad_();
  const Tensor plain = fg::zeros({3}, kGpu);
  EXPECT_THROW(plain.copy_(w), fg::Error);
  EXPECT_THROW(plain.add_(w), fg::Error);
  EXPECT_NO_THROW(plain.copy_(w.detach()));
}

TEST(InPlaceAutograd, AllowedWithoutGradMode) {
  SKIP_IF_NO_CUDA();
  // An SGD step: w -= lr * grad, then clear the gradient.
  const Tensor w = fg::ones({3}, kGpu).requires_grad_();
  fg::sum(w * w).backward();
  {
    const fg::NoGradGuard no_grad;
    w.sub_(w.grad() * 0.25f);
    w.grad().zero_();
  }
  EXPECT_EQ(w.to_vector(), (std::vector<float>{0.5f, 0.5f, 0.5f}));
  EXPECT_EQ(w.grad().to_vector(), (std::vector<float>{0, 0, 0}));
  EXPECT_TRUE(w.requires_grad());
  EXPECT_TRUE(w.is_leaf());
}

TEST(InPlaceAutograd, GradientsAndDetachedTensorsAreWritable) {
  SKIP_IF_NO_CUDA();
  const Tensor w = fg::ones({2}, kGpu).requires_grad_();
  fg::sum(w).backward();
  EXPECT_NO_THROW(w.grad().mul_(2.0));  // grad() doesn't require grad
  EXPECT_EQ(w.grad().to_vector(), (std::vector<float>{2, 2}));
  EXPECT_NO_THROW(w.detach().fill_(3.0));  // shares w's memory: the caller's responsibility
  EXPECT_EQ(w.to_vector(), (std::vector<float>{3, 3}));
}

}  // namespace
