#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "flashgrad/nn/linear.h"
#include "flashgrad/nn/loss.h"
#include "flashgrad/optim/adam.h"
#include "reference.h"

// optim::Adam (and AdamW through decoupled_weight_decay), against a host reference in double.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
namespace nn = flashgrad::nn;
namespace optim = flashgrad::optim;
using fg::Tensor;

// The parameter values after each step of Adam, starting from p with the given gradients.
std::vector<std::vector<double>> adam_ref(std::vector<double> p,
                                          const std::vector<std::vector<double>>& grads,
                                          const optim::AdamOptions& o) {
  std::vector<double> m(p.size(), 0.0), v(p.size(), 0.0);
  std::vector<std::vector<double>> history;
  for (std::size_t t = 1; t <= grads.size(); ++t) {
    const double c1 = 1.0 - std::pow(static_cast<double>(o.beta1), static_cast<double>(t));
    const double c2 = 1.0 - std::pow(static_cast<double>(o.beta2), static_cast<double>(t));
    for (std::size_t i = 0; i < p.size(); ++i) {
      double g = grads[t - 1][i];
      if (o.weight_decay != 0.0f) {
        if (o.decoupled_weight_decay) {
          p[i] *= 1.0 - static_cast<double>(o.lr) * o.weight_decay;
        } else {
          g += static_cast<double>(o.weight_decay) * p[i];
        }
      }
      m[i] = o.beta1 * m[i] + (1.0 - o.beta1) * g;
      v[i] = o.beta2 * v[i] + (1.0 - o.beta2) * g * g;
      p[i] -= o.lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + o.eps);
    }
    history.push_back(p);
  }
  return history;
}

// Runs Adam on the GPU with the gradients and compares every step with adam_ref.
void check_against_reference(const optim::AdamOptions& options) {
  const std::vector<float> start = {1.0f, -0.5f, 2.0f};
  const std::vector<std::vector<double>> grads = {
      {0.1, -0.2, 0.3}, {0.5, 0.1, -0.4}, {-0.3, 0.2, 0.0}, {0.05, -0.05, 0.2}};
  const Tensor p = fg::tensor(start, fg::Device::cuda()).requires_grad_();
  optim::Adam adam({p}, options);
  const auto expected = adam_ref(std::vector<double>(start.begin(), start.end()), grads, options);
  for (std::size_t t = 0; t < grads.size(); ++t) {
    const std::vector<float> g(grads[t].begin(), grads[t].end());
    p.set_grad(fg::tensor(g, fg::Device::cuda()));
    adam.step();
    const std::vector<float> e(expected[t].begin(), expected[t].end());
    EXPECT_TRUE(ref::close(p.to_vector(), e, 1e-5, 1e-6)) << "step " << t + 1;
  }
  EXPECT_EQ(adam.steps(), 4);
}

TEST(Adam, RejectsInvalidOptions) {
  const std::vector<Tensor> params = {fg::ones({2}).requires_grad_()};
  EXPECT_NO_THROW(optim::Adam(params, {}));
  EXPECT_THROW(optim::Adam(params, {.lr = 0.0f}), fg::Error);
  EXPECT_THROW(optim::Adam(params, {.beta1 = 1.0f}), fg::Error);
  EXPECT_THROW(optim::Adam(params, {.beta2 = -0.1f}), fg::Error);
  EXPECT_THROW(optim::Adam(params, {.eps = 0.0f}), fg::Error);
  EXPECT_THROW(optim::Adam(params, {.weight_decay = -0.1f}), fg::Error);
}

TEST(Adam, FirstStepMovesEachParameterByAboutLr) {
  SKIP_IF_NO_CUDA();
  // On step 1, m_hat = g and v_hat = g^2, so the update is lr * g / (|g| + eps) ~ lr * sign(g).
  const Tensor p = fg::tensor({1.0f, 1.0f, 1.0f}, fg::Device::cuda()).requires_grad_();
  p.set_grad(fg::tensor({0.5f, -2.0f, 0.001f}, fg::Device::cuda()));
  optim::Adam adam({p}, {.lr = 0.1f});
  adam.step();
  EXPECT_TRUE(ref::close(p.to_vector(), {0.9f, 1.1f, 0.9f}, 1e-5, 1e-6));
}

TEST(Adam, ZeroGradientAndStateLeaveTheParameterUnchanged) {
  SKIP_IF_NO_CUDA();
  // sqrt(0) must be 0, not NaN.
  const Tensor p = fg::tensor({1.0f, -3.0f}, fg::Device::cuda()).requires_grad_();
  p.set_grad(fg::zeros({2}, {.device = fg::Device::cuda()}));
  optim::Adam adam({p}, {});
  adam.step();
  EXPECT_EQ(p.to_vector(), std::vector<float>({1.0f, -3.0f}));
}

TEST(Adam, MatchesReference) {
  SKIP_IF_NO_CUDA();
  check_against_reference({.lr = 0.01f});
}

TEST(Adam, L2WeightDecayMatchesReference) {
  SKIP_IF_NO_CUDA();
  check_against_reference({.lr = 0.01f, .weight_decay = 0.1f});
}

TEST(Adam, DecoupledWeightDecayMatchesReference) {
  SKIP_IF_NO_CUDA();
  check_against_reference({.lr = 0.01f, .weight_decay = 0.1f, .decoupled_weight_decay = true});
}

TEST(Adam, SkipsParametersWithoutAGradientAndKeepsGradients) {
  SKIP_IF_NO_CUDA();
  const Tensor with = fg::tensor({1.0f}, fg::Device::cuda()).requires_grad_();
  with.set_grad(fg::tensor({1.0f}, fg::Device::cuda()));
  const Tensor without = fg::tensor({5.0f}, fg::Device::cuda()).requires_grad_();
  optim::Adam adam({with, without}, {.lr = 0.1f});
  adam.step();
  EXPECT_EQ(without.to_vector(), std::vector<float>({5.0f}));
  EXPECT_EQ(with.grad().to_vector(), std::vector<float>({1.0f}));
  EXPECT_TRUE(with.is_leaf());
}

TEST(Adam, TrainsALinearRegression) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  fg::manual_seed(5);
  nn::Linear fc(3, 1, {.device = gpu});
  const Tensor x = ref::random_cuda({64, 3}, 10);
  const Tensor target = fg::matmul(x, fg::tensor({2.0f, -1.0f, 0.5f}, {3, 1}, gpu)) + 0.3f;
  optim::Adam adam(fc.parameters(), {.lr = 0.05f});
  float first_loss = 0.0f, loss_value = 0.0f;
  for (int step = 0; step < 300; ++step) {
    const Tensor loss = nn::mse_loss(fc(x), target);
    loss_value = loss.item();
    if (step == 0) first_loss = loss_value;
    adam.zero_grad();
    loss.backward();
    adam.step();
  }
  EXPECT_LT(loss_value, 0.01f * first_loss);
}

}  // namespace
