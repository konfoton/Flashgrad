#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "flashgrad/nn/activation.h"
#include "flashgrad/nn/dropout.h"
#include "flashgrad/nn/sequential.h"
#include "reference.h"

// nn::Dropout: identity in evaluation mode, a scaled random mask in training mode.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
namespace nn = flashgrad::nn;
using fg::Tensor;

TEST(Dropout, Construction) {
  const nn::Dropout d;
  EXPECT_EQ(d.p(), 0.5f);
  EXPECT_TRUE(d.parameters().empty());
  EXPECT_EQ(nn::Dropout(0.2f).p(), 0.2f);
  EXPECT_NO_THROW(nn::Dropout(0.0f));
  EXPECT_THROW(nn::Dropout(-0.1f), fg::Error);
  EXPECT_THROW(nn::Dropout(1.0f), fg::Error);
  EXPECT_THROW(nn::Dropout(1.5f), fg::Error);
}

TEST(Dropout, EvaluationModeReturnsTheInput) {
  nn::Dropout d(0.5f);
  d.eval();
  const Tensor x = fg::ones({4, 4});
  EXPECT_TRUE(d(x).is_same(x));
}

TEST(Dropout, ZeroProbabilityReturnsTheInput) {
  nn::Dropout d(0.0f);
  const Tensor x = fg::ones({4, 4});
  EXPECT_TRUE(d(x).is_same(x));
}

TEST(Dropout, TrainingZeroesAboutPAndScalesTheRest) {
  SKIP_IF_NO_CUDA();
  fg::manual_seed(1);
  constexpr float p = 0.3f;
  nn::Dropout d(p);
  const Tensor x = fg::ones({100000}, {.device = fg::Device::cuda()});
  const Tensor y = d(x);
  EXPECT_EQ(y.shape(), x.shape());
  std::size_t zeros = 0;
  double sum = 0.0;
  for (const float v : y.to_vector()) {
    if (v == 0.0f) {
      ++zeros;
    } else {
      EXPECT_NEAR(v, 1.0f / (1.0f - p), 1e-6);
    }
    sum += v;
  }
  EXPECT_NEAR(static_cast<double>(zeros) / 100000.0, p, 0.01);
  EXPECT_NEAR(sum / 100000.0, 1.0, 0.02);  // the expected value is kept
}

TEST(Dropout, NewMaskOnEveryCall) {
  SKIP_IF_NO_CUDA();
  nn::Dropout d(0.5f);
  const Tensor x = fg::ones({1000}, {.device = fg::Device::cuda()});
  EXPECT_NE(d(x).to_vector(), d(x).to_vector());
}

TEST(Dropout, GradientFollowsTheMask) {
  SKIP_IF_NO_CUDA();
  constexpr float p = 0.5f;
  nn::Dropout d(p);
  const Tensor x = ref::random_cuda({1000}, 2).requires_grad_();
  const Tensor y = d(x);
  fg::sum(y).backward();
  const std::vector<float> out = y.to_vector();
  const std::vector<float> grad = x.grad().to_vector();
  for (std::size_t i = 0; i < out.size(); ++i) {
    const float expected = out[i] == 0.0f ? 0.0f : 1.0f / (1.0f - p);
    EXPECT_NEAR(grad[i], expected, 1e-6) << "element " << i;
  }
}

TEST(Dropout, FollowsTheModeOfItsParent) {
  nn::Sequential seq{std::make_shared<nn::ReLU>(), std::make_shared<nn::Dropout>(0.5f)};
  seq.eval();
  EXPECT_FALSE(seq[1]->is_training());
  const Tensor x = fg::ones({3});
  EXPECT_TRUE((*seq[1])(x).is_same(x));
}

}  // namespace
