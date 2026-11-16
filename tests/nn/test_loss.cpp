#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "flashgrad/nn/loss.h"
#include "reference.h"

// Loss functions: values against host references in double, and their gradients.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
namespace nn = flashgrad::nn;
using fg::Tensor;

// ---- mse_loss -------------------------------------------------------------------------

double mse_ref(const std::vector<float>& a, const std::vector<float>& b) {
  double total = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    total += d * d;
  }
  return total / static_cast<double>(a.size());
}

TEST(MseLoss, MatchesReference) {
  SKIP_IF_NO_CUDA();
  for (const fg::Shape& shape : {fg::Shape{7}, fg::Shape{4, 5}, fg::Shape{2, 3, 4}}) {
    const Tensor a = ref::random_cuda(shape, 1);
    const Tensor b = ref::random_cuda(shape, 2);
    const Tensor loss = nn::mse_loss(a, b);
    EXPECT_EQ(loss.dim(), 0);
    EXPECT_NEAR(loss.item(), mse_ref(a.to_vector(), b.to_vector()), 1e-5);
  }
}

TEST(MseLoss, IsZeroForEqualInputs) {
  SKIP_IF_NO_CUDA();
  const Tensor a = ref::random_cuda({3, 3}, 3);
  EXPECT_EQ(nn::mse_loss(a, a).item(), 0.0f);
}

TEST(MseLoss, Gradient) {
  SKIP_IF_NO_CUDA();
  // d/da mean((a - b)^2) = 2 (a - b) / N
  const Tensor a = ref::random_cuda({4, 5}, 4).requires_grad_();
  const Tensor b = ref::random_cuda({4, 5}, 5);
  nn::mse_loss(a, b).backward();
  const std::vector<float> av = a.to_vector(), bv = b.to_vector();
  std::vector<float> expected(av.size());
  for (std::size_t i = 0; i < av.size(); ++i) expected[i] = 2.0f * (av[i] - bv[i]) / 20.0f;
  EXPECT_TRUE(ref::close(a.grad().to_vector(), expected, 1e-5, 1e-6));
}

TEST(MseLoss, RejectsDifferentShapes) {
  SKIP_IF_NO_CUDA();
  const fg::TensorOptions gpu{.device = fg::Device::cuda()};
  EXPECT_THROW(nn::mse_loss(fg::ones({4, 1}, gpu), fg::ones({4}, gpu)), fg::Error);
  EXPECT_THROW(nn::mse_loss(fg::ones({3, 2}, gpu), fg::ones({2, 3}, gpu)), fg::Error);
}

// ---- cross_entropy --------------------------------------------------------------------

// Mean over rows of logsumexp(row) - row[target], in double.
double cross_entropy_ref(const std::vector<float>& logits, const std::vector<float>& target,
                         std::size_t classes) {
  const std::size_t batch = target.size();
  double total = 0.0;
  for (std::size_t n = 0; n < batch; ++n) {
    const float* row = &logits[n * classes];
    double max = row[0];
    for (std::size_t c = 1; c < classes; ++c) max = std::max(max, static_cast<double>(row[c]));
    double sum = 0.0;
    for (std::size_t c = 0; c < classes; ++c) sum += std::exp(row[c] - max);
    total += max + std::log(sum) - row[static_cast<std::size_t>(target[n])];
  }
  return total / static_cast<double>(batch);
}

TEST(CrossEntropy, MatchesReference) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  const Tensor logits = ref::random_cuda({6, 4}, 6);
  const std::vector<float> labels = {0, 3, 1, 1, 2, 0};
  const Tensor target = fg::tensor(labels, gpu);
  const Tensor loss = nn::cross_entropy(logits, target);
  EXPECT_EQ(loss.dim(), 0);
  EXPECT_NEAR(loss.item(), cross_entropy_ref(logits.to_vector(), labels, 4), 1e-5);
}

TEST(CrossEntropy, UniformLogitsGiveLogOfClassCount) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  const Tensor logits = fg::zeros({3, 10}, {.device = gpu});
  const Tensor target = fg::tensor({2.0f, 7.0f, 9.0f}, gpu);
  EXPECT_NEAR(nn::cross_entropy(logits, target).item(), std::log(10.0), 1e-6);
}

TEST(CrossEntropy, LargeLogitsStayFinite) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  const Tensor logits = fg::tensor({1000.0f, -1000.0f, 0.0f, -1000.0f, 1000.0f, 0.0f}, {2, 3}, gpu);
  const Tensor target = fg::tensor({0.0f, 0.0f}, gpu);
  // Row 0 is confidently right (loss ~0), row 1 confidently wrong (loss ~2000).
  EXPECT_NEAR(nn::cross_entropy(logits, target).item(), 1000.0, 1e-3);
}

TEST(CrossEntropy, GradientIsSoftmaxMinusOneHotOverBatch) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  const Tensor logits = ref::random_cuda({4, 5}, 7).requires_grad_();
  const std::vector<float> labels = {1, 0, 4, 2};
  nn::cross_entropy(logits, fg::tensor(labels, gpu)).backward();

  const std::vector<float> z = logits.to_vector();
  std::vector<float> expected(z.size());
  for (std::size_t n = 0; n < 4; ++n) {
    double sum = 0.0;
    for (std::size_t c = 0; c < 5; ++c) sum += std::exp(static_cast<double>(z[n * 5 + c]));
    for (std::size_t c = 0; c < 5; ++c) {
      const double p = std::exp(static_cast<double>(z[n * 5 + c])) / sum;
      const double one_hot = c == static_cast<std::size_t>(labels[n]) ? 1.0 : 0.0;
      expected[n * 5 + c] = static_cast<float>((p - one_hot) / 4.0);
    }
  }
  EXPECT_TRUE(ref::close(logits.grad().to_vector(), expected, 1e-5, 1e-6));
}

TEST(CrossEntropy, RejectsBadShapes) {
  SKIP_IF_NO_CUDA();
  const fg::TensorOptions gpu{.device = fg::Device::cuda()};
  EXPECT_THROW(nn::cross_entropy(fg::zeros({4}, gpu), fg::zeros({4}, gpu)), fg::Error);
  EXPECT_THROW(nn::cross_entropy(fg::zeros({4, 3}, gpu), fg::zeros({4, 1}, gpu)), fg::Error);
  EXPECT_THROW(nn::cross_entropy(fg::zeros({4, 3}, gpu), fg::zeros({5}, gpu)), fg::Error);
}

}  // namespace
