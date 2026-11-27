#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
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

// Runs Adam on the GPU with the gradients and compares every step with adam_ref, both with
// the fused kernel and with the unfused composite ops.
void check_against_reference(optim::AdamOptions options) {
  const std::vector<float> start = {1.0f, -0.5f, 2.0f};
  const std::vector<std::vector<double>> grads = {
      {0.1, -0.2, 0.3}, {0.5, 0.1, -0.4}, {-0.3, 0.2, 0.0}, {0.05, -0.05, 0.2}};
  const auto expected = adam_ref(std::vector<double>(start.begin(), start.end()), grads, options);
  for (const bool fused : {true, false}) {
    options.fused = fused;
    const Tensor p = fg::tensor(start, fg::Device::cuda()).requires_grad_();
    optim::Adam adam({p}, options);
    for (std::size_t t = 0; t < grads.size(); ++t) {
      const std::vector<float> g(grads[t].begin(), grads[t].end());
      p.set_grad(fg::tensor(g, fg::Device::cuda()));
      adam.step();
      const std::vector<float> e(expected[t].begin(), expected[t].end());
      EXPECT_TRUE(ref::close(p.to_vector(), e, 1e-5, 1e-6))
          << "step " << t + 1 << (fused ? " (fused)" : " (unfused)");
    }
    EXPECT_EQ(adam.steps(), 4);
  }
}

// Runs `steps` Adam steps on `params` with random gradients (set with `make_grad`, which gets
// the parameter and a contiguous random gradient of its shape) and compares every parameter
// with adam_ref afterwards.
template <typename MakeGrad>
void check_parameters(const std::vector<Tensor>& params, const optim::AdamOptions& options,
                      int steps, MakeGrad make_grad) {
  std::vector<std::vector<double>> start;
  for (const Tensor& p : params) {
    const std::vector<float> values = p.to_vector();
    start.emplace_back(values.begin(), values.end());
  }
  std::vector<std::vector<std::vector<double>>> grads(params.size());  // [param][step][i]
  optim::Adam adam(params, options);
  std::uint64_t seed = 100;
  for (int t = 0; t < steps; ++t) {
    for (std::size_t i = 0; i < params.size(); ++i) {
      const Tensor g = ref::random_cuda(params[i].shape(), seed++);
      const std::vector<float> values = g.to_vector();
      grads[i].emplace_back(values.begin(), values.end());
      params[i].set_grad(make_grad(params[i], g));
    }
    adam.step();
  }
  for (std::size_t i = 0; i < params.size(); ++i) {
    const std::vector<double> expected = adam_ref(start[i], grads[i], options).back();
    EXPECT_TRUE(ref::close(params[i].to_vector(),
                           std::vector<float>(expected.begin(), expected.end()), 1e-5, 1e-6))
        << "parameter " << i << " " << fg::describe(params[i]);
  }
}

Tensor same_grad(const Tensor& /*param*/, const Tensor& grad) { return grad; }

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

TEST(Adam, FusedHandlesManyParametersOfAwkwardSizes) {
  SKIP_IF_NO_CUDA();
  // More than one launch (64 tensors each), empty and size-1 tensors, and sizes around the
  // 4096-element chunk and the float4 width.
  std::vector<fg::Shape> shapes = {{0},    {1},    {3},    {4},    {5},      {4095},
                                   {4096}, {4097}, {8193}, {2, 0}, {33, 65}, {7, 3, 5}};
  for (std::int64_t i = 0; i < 70; ++i) shapes.push_back({1 + (i * 37) % 300});
  std::vector<Tensor> params;
  for (std::size_t i = 0; i < shapes.size(); ++i)
    params.push_back(ref::random_cuda(shapes[i], i).requires_grad_());
  check_parameters(params, {.lr = 0.01f}, 3, same_grad);
}

TEST(Adam, FusedAdamWHandlesManyParameters) {
  SKIP_IF_NO_CUDA();
  std::vector<Tensor> params;
  for (std::int64_t i = 0; i < 80; ++i)
    params.push_back(ref::random_cuda({1 + (i * 911) % 5000}, i).requires_grad_());
  check_parameters(params, {.lr = 0.01f, .weight_decay = 0.1f, .decoupled_weight_decay = true}, 3,
                   same_grad);
}

TEST(Adam, FusedTakesNonContiguousGradients) {
  SKIP_IF_NO_CUDA();
  const Tensor p = ref::random_cuda({5, 7}, 1).requires_grad_();
  check_parameters({p}, {.lr = 0.01f, .weight_decay = 0.1f}, 3, [](const Tensor&, const Tensor& g) {
    const Tensor transposed = g.t().contiguous().t();  // same values, strided
    EXPECT_FALSE(transposed.is_contiguous());
    return transposed;
  });
}

TEST(Adam, FusedTakesUnalignedParameters) {
  SKIP_IF_NO_CUDA();
  // Contiguous views starting one float in, so the kernel can't use float4 accesses.
  const Tensor base = ref::random_cuda({5000}, 2);
  const Tensor p = base.slice(0, 1).detach().requires_grad_();
  const Tensor q = base.slice(0, 1, 8).detach().requires_grad_();
  ASSERT_TRUE(p.is_contiguous());
  check_parameters({p, q}, {.lr = 0.01f}, 3, same_grad);
}

TEST(Adam, NonContiguousParametersFallBackToTheUnfusedStep) {
  SKIP_IF_NO_CUDA();
  const Tensor p = ref::random_cuda({6, 9}, 3).t().detach().requires_grad_();
  ASSERT_FALSE(p.is_contiguous());
  const Tensor q = ref::random_cuda({100}, 4).requires_grad_();
  check_parameters({p, q}, {.lr = 0.01f}, 3, same_grad);
}

TEST(Adam, FusedRejectsCpuParameters) {
  const Tensor p = fg::ones({2}).requires_grad_();
  p.set_grad(fg::ones({2}));
  optim::Adam adam({p}, {});
  EXPECT_THROW(adam.step(), fg::Error);
}

}  // namespace
