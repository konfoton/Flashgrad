#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "flashgrad/autograd/grad_mode.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "flashgrad/nn/linear.h"
#include "reference.h"

// nn::Linear: parameter shapes and initialization (on the CPU), forward and gradients (GPU).
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
namespace nn = flashgrad::nn;
using fg::Tensor;

std::vector<std::string> names(const nn::Module& module) {
  std::vector<std::string> out;
  for (const auto& [name, tensor] : module.named_parameters()) out.push_back(name);
  return out;
}

// ---- Construction ---------------------------------------------------------------------

TEST(Linear, ParameterShapesAndNames) {
  const nn::Linear fc(3, 4);
  EXPECT_EQ(fc.in_features(), 3);
  EXPECT_EQ(fc.out_features(), 4);
  EXPECT_EQ(fc.weight().shape(), fg::Shape({3, 4}));
  EXPECT_EQ(fc.bias().shape(), fg::Shape({4}));
  EXPECT_EQ(names(fc), (std::vector<std::string>{"weight", "bias"}));
  EXPECT_TRUE(fc.weight().requires_grad());
  EXPECT_TRUE(fc.bias().requires_grad());
  EXPECT_TRUE(fc.weight().device().is_cpu());
}

TEST(Linear, WithoutBias) {
  const nn::Linear fc(3, 4, {.bias = false});
  EXPECT_FALSE(fc.bias().defined());
  EXPECT_EQ(names(fc), (std::vector<std::string>{"weight"}));
}

TEST(Linear, InitializationIsUniformOnPlusMinusOneOverSqrtIn) {
  fg::manual_seed(1);
  const nn::Linear fc(100, 200);
  const double k = 1.0 / std::sqrt(100.0);
  for (const Tensor& t : {fc.weight(), fc.bias()}) {
    const std::vector<float> values = t.to_vector();
    double sum = 0.0, sum_sq = 0.0;
    for (const float v : values) {
      EXPECT_GE(v, -k);
      EXPECT_LE(v, k);
      sum += v;
      sum_sq += static_cast<double>(v) * v;
    }
    const double n = static_cast<double>(values.size());
    const double mean = sum / n;
    const double std_dev = std::sqrt(sum_sq / n - mean * mean);
    EXPECT_NEAR(mean, 0.0, 0.2 * k);
    EXPECT_NEAR(std_dev, k / std::sqrt(3.0), 0.1 * k);  // std of U(-k, k)
  }
}

TEST(Linear, ManualSeedMakesInitializationReproducible) {
  fg::manual_seed(42);
  const nn::Linear a(5, 6);
  fg::manual_seed(42);
  const nn::Linear b(5, 6);
  const nn::Linear c(5, 6);  // the generator moved on
  EXPECT_EQ(a.weight().to_vector(), b.weight().to_vector());
  EXPECT_EQ(a.bias().to_vector(), b.bias().to_vector());
  EXPECT_NE(a.weight().to_vector(), c.weight().to_vector());
}

TEST(Linear, RejectsNonPositiveSizes) {
  EXPECT_THROW(nn::Linear(0, 4), fg::Error);
  EXPECT_THROW(nn::Linear(3, 0), fg::Error);
  EXPECT_THROW(nn::Linear(-1, 4), fg::Error);
}

TEST(Linear, CreatesParametersOnTheRequestedDevice) {
  SKIP_IF_NO_CUDA();
  const nn::Linear fc(3, 4, {.device = fg::Device::cuda()});
  EXPECT_TRUE(fc.weight().device().is_cuda());
  EXPECT_TRUE(fc.bias().device().is_cuda());
}

// ---- Forward --------------------------------------------------------------------------

// x @ weight + bias on the host, for x [m, in].
std::vector<float> linear_ref(const Tensor& x, const nn::Linear& fc) {
  const std::int64_t m = x.size(0), in = fc.in_features(), out = fc.out_features();
  std::vector<float> y = ref::matmul(x.to_vector(), fc.weight().to_vector(), m, in, out);
  if (fc.bias().defined()) {
    const std::vector<float> b = fc.bias().to_vector();
    for (std::size_t i = 0; i < y.size(); ++i) y[i] += b[i % b.size()];
  }
  return y;
}

TEST(Linear, ForwardMatchesReference) {
  SKIP_IF_NO_CUDA();
  nn::Linear fc(7, 5, {.device = fg::Device::cuda()});
  for (const std::int64_t batch : {1, 3, 64}) {
    const Tensor x = ref::random_cuda({batch, 7}, 2);
    const Tensor y = fc(x);
    EXPECT_EQ(y.shape(), fg::Shape({batch, 5}));
    EXPECT_TRUE(ref::close(y.to_vector(), linear_ref(x, fc), 1e-5, 1e-5)) << "batch " << batch;
  }
}

TEST(Linear, ForwardWithoutBias) {
  SKIP_IF_NO_CUDA();
  nn::Linear fc(4, 3, {.bias = false, .device = fg::Device::cuda()});
  const Tensor x = ref::random_cuda({6, 4}, 3);
  EXPECT_TRUE(ref::close(fc(x).to_vector(), linear_ref(x, fc), 1e-5, 1e-5));
}

TEST(Linear, ForwardRejectsWrongInputShapes) {
  SKIP_IF_NO_CUDA();
  nn::Linear fc(4, 3, {.device = fg::Device::cuda()});
  EXPECT_THROW(fc(ref::random_cuda({6, 5}, 4)), fg::Error);     // wrong feature count
  EXPECT_THROW(fc(ref::random_cuda({4}, 5)), fg::Error);        // 1-D
  EXPECT_THROW(fc(ref::random_cuda({2, 3, 4}, 6)), fg::Error);  // 3-D
}

// ---- Gradients ------------------------------------------------------------------------

TEST(Linear, GradientsOfTheSum) {
  SKIP_IF_NO_CUDA();
  // For L = sum(x @ W + b): dL/dW[i][j] = sum_n x[n][i] and dL/db[j] = batch.
  nn::Linear fc(3, 2, {.device = fg::Device::cuda()});
  const Tensor x = ref::random_cuda({5, 3}, 7);
  fg::sum(fc(x)).backward();

  const std::vector<float> xs = x.to_vector();
  std::vector<float> expected_w(6, 0.0f);
  for (std::size_t n = 0; n < 5; ++n) {
    for (std::size_t i = 0; i < 3; ++i) {
      expected_w[i * 2] += xs[n * 3 + i];
      expected_w[i * 2 + 1] += xs[n * 3 + i];
    }
  }
  EXPECT_TRUE(ref::close(fc.weight().grad().to_vector(), expected_w, 1e-5, 1e-5));
  EXPECT_TRUE(ref::close(fc.bias().grad().to_vector(), {5.0f, 5.0f}, 1e-6, 1e-6));
}

TEST(Linear, GradientDescentFitsALinearMap) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  fg::manual_seed(3);
  nn::Linear fc(3, 2, {.device = gpu});
  const Tensor true_w = fg::tensor({1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 0.0f}, {3, 2}, gpu);
  const Tensor true_b = fg::tensor({0.25f, -0.75f}, gpu);
  const Tensor x = ref::random_cuda({64, 3}, 8);
  const Tensor target = fg::matmul(x, true_w) + true_b;

  constexpr float kLearningRate = 0.5f;
  float loss_value = 0.0f;
  for (int step = 0; step < 300; ++step) {
    const Tensor diff = fc(x) - target;
    const Tensor loss = fg::mean(diff * diff);
    loss_value = loss.item();
    fc.zero_grad();
    loss.backward();
    const fg::NoGradGuard no_grad;
    for (const Tensor& p : fc.parameters()) p.sub_(p.grad() * kLearningRate);
  }
  EXPECT_LT(loss_value, 1e-5f);
  EXPECT_TRUE(ref::close(fc.weight().to_vector(), true_w.to_vector(), 1e-3, 1e-3));
  EXPECT_TRUE(ref::close(fc.bias().to_vector(), true_b.to_vector(), 1e-3, 1e-3));
}

}  // namespace
