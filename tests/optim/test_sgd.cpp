#include <gtest/gtest.h>

#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "flashgrad/nn/linear.h"
#include "flashgrad/nn/loss.h"
#include "flashgrad/optim/sgd.h"
#include "reference.h"

// optim::Optimizer and optim::SGD. Construction is checked on the CPU; steps need a GPU.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
namespace nn = flashgrad::nn;
namespace optim = flashgrad::optim;
using fg::Tensor;

// A CUDA leaf that requires grad, with the given values and gradient.
Tensor parameter(const std::vector<float>& values, const std::vector<float>& grad) {
  const fg::Device gpu = fg::Device::cuda();
  const Tensor p = fg::tensor(values, gpu).requires_grad_();
  p.set_grad(fg::tensor(grad, gpu));
  return p;
}

// ---- Construction ---------------------------------------------------------------------

TEST(Optimizer, KeepsHandlesToTheModelsParameters) {
  const nn::Linear fc(2, 3);
  const optim::SGD sgd(fc.parameters(), {});
  ASSERT_EQ(sgd.parameters().size(), 2u);
  EXPECT_TRUE(sgd.parameters()[0].is_same(fc.weight()));
  EXPECT_TRUE(sgd.parameters()[1].is_same(fc.bias()));
}

TEST(Optimizer, RejectsParametersThatDontRequireGrad) {
  EXPECT_THROW(optim::SGD({fg::ones({2})}, {}), fg::Error);
  EXPECT_THROW(optim::SGD({Tensor()}, {}), fg::Error);
}

TEST(SGD, RejectsInvalidOptions) {
  const std::vector<Tensor> params = {fg::ones({2}).requires_grad_()};
  EXPECT_NO_THROW(optim::SGD(params, {.lr = 0.1f, .momentum = 0.9f, .weight_decay = 0.01f}));
  EXPECT_THROW(optim::SGD(params, {.lr = 0.0f}), fg::Error);
  EXPECT_THROW(optim::SGD(params, {.lr = -0.1f}), fg::Error);
  EXPECT_THROW(optim::SGD(params, {.momentum = -0.1f}), fg::Error);
  EXPECT_THROW(optim::SGD(params, {.momentum = 1.0f}), fg::Error);
  EXPECT_THROW(optim::SGD(params, {.weight_decay = -1.0f}), fg::Error);
}

// ---- Steps ----------------------------------------------------------------------------

TEST(Optimizer, ZeroGradClearsEveryGradient) {
  SKIP_IF_NO_CUDA();
  const Tensor a = parameter({1, 2}, {3, 4});
  const Tensor b = parameter({5}, {6});
  optim::SGD sgd({a, b}, {});
  sgd.zero_grad();
  EXPECT_FALSE(a.grad().defined());
  EXPECT_FALSE(b.grad().defined());
}

TEST(SGD, PlainStep) {
  SKIP_IF_NO_CUDA();
  const Tensor p = parameter({1.0f, 2.0f, 3.0f}, {0.5f, -1.0f, 2.0f});
  optim::SGD sgd({p}, {.lr = 0.1f});
  sgd.step();
  EXPECT_TRUE(ref::close(p.to_vector(), {0.95f, 2.1f, 2.8f}, 1e-6, 1e-6));
}

TEST(SGD, StepKeepsParametersLeavesAndGradientsUnchanged) {
  SKIP_IF_NO_CUDA();
  const Tensor p = parameter({1.0f, 2.0f}, {1.0f, 1.0f});
  optim::SGD sgd({p}, {.lr = 0.1f, .momentum = 0.9f});
  sgd.step();
  sgd.step();
  EXPECT_TRUE(p.is_leaf());
  EXPECT_TRUE(p.requires_grad());
  EXPECT_EQ(p.grad().to_vector(), std::vector<float>({1.0f, 1.0f}));
}

TEST(SGD, SkipsParametersWithoutAGradient) {
  SKIP_IF_NO_CUDA();
  const Tensor with = parameter({1.0f}, {1.0f});
  const Tensor without = fg::tensor({5.0f}, fg::Device::cuda()).requires_grad_();
  optim::SGD sgd({with, without}, {.lr = 0.5f});
  sgd.step();
  EXPECT_TRUE(ref::close(with.to_vector(), {0.5f}, 1e-6, 1e-6));
  EXPECT_EQ(without.to_vector(), std::vector<float>({5.0f}));
}

TEST(SGD, WeightDecay) {
  SKIP_IF_NO_CUDA();
  // With a zero gradient, g = weight_decay * p, so p shrinks by a factor 1 - lr * weight_decay.
  const Tensor p = parameter({1.0f, -2.0f}, {0.0f, 0.0f});
  optim::SGD sgd({p}, {.lr = 0.1f, .weight_decay = 0.5f});
  sgd.step();
  EXPECT_TRUE(ref::close(p.to_vector(), {0.95f, -1.9f}, 1e-6, 1e-6));
}

TEST(SGD, MomentumAccumulates) {
  SKIP_IF_NO_CUDA();
  // Constant gradient 1, lr 0.1, momentum 0.9: v = 1, 1.9, 2.71.
  const Tensor p = parameter({0.0f}, {1.0f});
  optim::SGD sgd({p}, {.lr = 0.1f, .momentum = 0.9f});
  const std::vector<float> expected = {-0.1f, -0.29f, -0.561f};
  for (const float e : expected) {
    sgd.step();
    EXPECT_NEAR(p.to_vector()[0], e, 1e-6);
  }
}

TEST(SGD, TrainsALinearRegression) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  fg::manual_seed(4);
  nn::Linear fc(3, 1, {.device = gpu});
  const Tensor x = ref::random_cuda({64, 3}, 9);
  const Tensor target = fg::matmul(x, fg::tensor({2.0f, -1.0f, 0.5f}, {3, 1}, gpu)) + 0.3f;
  optim::SGD sgd(fc.parameters(), {.lr = 0.05f, .momentum = 0.9f});
  float loss_value = 0.0f;
  for (int step = 0; step < 200; ++step) {
    const Tensor loss = nn::mse_loss(fc(x), target);
    loss_value = loss.item();
    sgd.zero_grad();
    loss.backward();
    sgd.step();
  }
  EXPECT_LT(loss_value, 1e-4f);
}

}  // namespace
