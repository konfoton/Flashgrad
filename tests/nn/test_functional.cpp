#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <numbers>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "flashgrad/nn/functional.h"
#include "gradcheck.h"
#include "reference.h"

// Activation functions from nn/functional.h: values against double references on the host,
// gradients against finite differences.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
namespace F = flashgrad::nn::functional;
using fg::Tensor;
using gradcheck::Values;

using UnaryOp = std::function<Tensor(const Tensor&)>;
using UnaryRef = std::function<double(double)>;

// ---- Host references, in double --------------------------------------------------------

double silu_ref(double x) { return x / (1.0 + std::exp(-x)); }

double tanh_ref(double x) { return std::tanh(x); }

double gelu_ref(double x) {
  const double k = std::sqrt(2.0 / std::numbers::pi);
  return 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
}

UnaryRef leaky_relu_ref(float negative_slope) {
  const double slope = negative_slope;
  return [slope](double x) { return x > 0.0 ? x : slope * x; };
}

// softmax of row-major `a` of `shape` along a non-negative `dim`, in double.
Values softmax_ref(const Values& a, const fg::Shape& shape, std::int64_t dim) {
  Values out = ref::log_softmax(a, shape, dim);
  for (double& v : out) v = std::exp(v);
  return out;
}

// ---- Helpers ---------------------------------------------------------------------------

// Inputs of several shapes and layouts; standard normal values times `scale`.
std::vector<Tensor> test_inputs(float scale) {
  const auto scaled = [scale](float v) { return scale * v; };
  return {
      ref::random_cuda({}, 1, scaled),
      ref::random_cuda({7}, 2, scaled),
      ref::random_cuda({3, 4, 5}, 3, scaled),
      ref::random_cuda({4, 6}, 4, scaled).t().slice(0, 1, 6, 2),  // non-contiguous
      ref::random_cuda({1, 4}, 5, scaled).expand({3, 4}),          // stride 0
      ref::random_cuda({0, 2}, 6, scaled),                         // empty
  };
}

// op(a) must match the elementwise reference on every test input.
void check_values(const UnaryOp& op, const UnaryRef& reference, float scale) {
  for (const Tensor& a : test_inputs(scale)) {
    const Tensor out = op(a);
    EXPECT_EQ(out.shape(), a.shape()) << fg::describe(a);
    EXPECT_TRUE(out.device().is_cuda()) << fg::describe(a);
    EXPECT_TRUE(ref::close(out.to_vector(), ref::unary(a.to_vector(), reference), 1e-5, 1e-6))
        << fg::describe(a);
  }
}

// The gradients of op must match finite differences of the reference on every test input.
void check_grads(const UnaryOp& op, const UnaryRef& reference, float scale) {
  const auto op_on_inputs = [&](const std::vector<Tensor>& in) { return op(in[0]); };
  const auto ref_on_inputs = [&](const std::vector<Values>& in) {
    return ref::unary(in[0], reference);
  };
  for (const Tensor& a : test_inputs(scale)) {
    EXPECT_TRUE(gradcheck::check(op_on_inputs, ref_on_inputs, {a})) << fg::describe(a);
  }
}

// op on huge inputs must give finite values close to `expected`, and finite gradients.
void check_large_inputs(const UnaryOp& op, const std::vector<float>& inputs,
                        const std::vector<float>& expected) {
  const Tensor x = fg::tensor(inputs, fg::Device::cuda()).requires_grad_();
  const Tensor out = op(x);
  EXPECT_TRUE(ref::close(out.to_vector(), expected, 1e-6, 1e-6));
  fg::sum(out).backward();
  for (const float g : x.grad().to_vector()) EXPECT_TRUE(std::isfinite(g)) << "gradient " << g;
}

// Keeps values at least 0.5 away from zero, where leaky_relu has a kink.
float away_from_zero(float v) { return v >= 0.0f ? v + 0.5f : v - 0.5f; }

// ---- silu ------------------------------------------------------------------------------

TEST(Silu, MatchesReference) {
  SKIP_IF_NO_CUDA();
  check_values(F::silu, silu_ref, 3.0f);
}

TEST(Silu, Gradients) {
  SKIP_IF_NO_CUDA();
  check_grads(F::silu, silu_ref, 2.0f);
}

// ---- leaky_relu ------------------------------------------------------------------------

TEST(LeakyRelu, DefaultSlopeIsOneHundredth) {
  SKIP_IF_NO_CUDA();
  check_values([](const Tensor& x) { return F::leaky_relu(x); }, leaky_relu_ref(0.01f), 1.0f);
}

TEST(LeakyRelu, CustomSlopes) {
  SKIP_IF_NO_CUDA();
  for (const float slope : {0.2f, 0.0f, 1.0f, 3.0f}) {
    check_values([slope](const Tensor& x) { return F::leaky_relu(x, slope); },
                 leaky_relu_ref(slope), 1.0f);
  }
}

TEST(LeakyRelu, Gradients) {
  SKIP_IF_NO_CUDA();
  for (const float slope : {0.01f, 0.2f}) {
    const auto op = [slope](const std::vector<Tensor>& in) { return F::leaky_relu(in[0], slope); };
    const auto reference = [slope](const std::vector<Values>& in) {
      return ref::unary(in[0], leaky_relu_ref(slope));
    };
    for (const fg::Shape& shape : {fg::Shape{7}, fg::Shape{3, 4}}) {
      const Tensor a = ref::random_cuda(shape, 7, away_from_zero);
      EXPECT_TRUE(gradcheck::check(op, reference, {a})) << "slope " << slope;
    }
  }
}

// ---- tanh ------------------------------------------------------------------------------

TEST(Tanh, MatchesReference) {
  SKIP_IF_NO_CUDA();
  check_values(F::tanh, tanh_ref, 3.0f);
}

TEST(Tanh, Gradients) {
  SKIP_IF_NO_CUDA();
  check_grads(F::tanh, tanh_ref, 1.5f);
}

TEST(Tanh, LargeInputsSaturate) {
  SKIP_IF_NO_CUDA();
  check_large_inputs(F::tanh, {-1000.0f, -100.0f, -20.0f, 0.0f, 20.0f, 100.0f, 1000.0f},
                     {-1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 1.0f});
}

// ---- softmax ---------------------------------------------------------------------------

TEST(Softmax, MatchesReferenceAlongEachDimension) {
  SKIP_IF_NO_CUDA();
  const Tensor contiguous = ref::random_cuda({3, 4, 5}, 8);
  const Tensor strided = ref::random_cuda({6, 7, 5}, 9).permute({2, 0, 1}).slice(1, 0, 6, 2);
  for (const Tensor& a : {contiguous, strided}) {
    for (const std::int64_t dim : {0, 1, 2, -1}) {
      const std::int64_t d = dim < 0 ? dim + a.dim() : dim;
      const std::vector<float> x = a.to_vector();
      const Values expected = softmax_ref(Values(x.begin(), x.end()), a.shape(), d);
      const Tensor out = F::softmax(a, dim);
      EXPECT_EQ(out.shape(), a.shape());
      EXPECT_TRUE(ref::close(out.to_vector(), std::vector<float>(expected.begin(), expected.end()),
                             1e-5, 1e-7))
          << "dim " << dim << " of " << fg::describe(a);
    }
  }
}

TEST(Softmax, SlicesSumToOne) {
  SKIP_IF_NO_CUDA();
  const Tensor probs = F::softmax(ref::random_cuda({8, 300}, 10) * 5.0f, 1);
  EXPECT_TRUE(ref::close(fg::sum(probs, 1).to_vector(), std::vector<float>(8, 1.0f), 1e-5, 1e-5));
}

TEST(Softmax, LargeInputsDontOverflow) {
  SKIP_IF_NO_CUDA();
  // exp(1000) overflows float32.
  const Tensor a = fg::tensor({1000.0f, 1001.0f, 1002.0f, -1000.0f, -1001.0f, -1002.0f}, {2, 3},
                              fg::Device::cuda());
  const std::vector<float> out = F::softmax(a, 1).to_vector();
  const std::vector<float> row = {0.09003057f, 0.24472847f, 0.66524096f};
  EXPECT_TRUE(ref::close(out, {row[0], row[1], row[2], row[2], row[1], row[0]}, 1e-5, 1e-7));
}

TEST(Softmax, Gradients) {
  SKIP_IF_NO_CUDA();
  const Tensor a = ref::random_cuda({3, 4, 5}, 11);
  for (const std::int64_t dim : {0, 1, 2, -1}) {
    const std::int64_t d = dim < 0 ? dim + a.dim() : dim;
    const auto op = [dim](const std::vector<Tensor>& in) { return F::softmax(in[0], dim); };
    const auto reference = [&](const std::vector<Values>& in) {
      return softmax_ref(in[0], a.shape(), d);
    };
    EXPECT_TRUE(gradcheck::check(op, reference, {a})) << "dim " << dim;
  }
}

TEST(Softmax, RejectsInvalidDims) {
  SKIP_IF_NO_CUDA();
  const fg::TensorOptions gpu{.device = fg::Device::cuda()};
  EXPECT_THROW(F::softmax(fg::ones({2, 3}, gpu), 2), fg::Error);
  EXPECT_THROW(F::softmax(fg::ones({2, 3}, gpu), -3), fg::Error);
}

// ---- gelu ------------------------------------------------------------------------------

TEST(Gelu, MatchesReference) {
  SKIP_IF_NO_CUDA();
  check_values(F::gelu, gelu_ref, 3.0f);
}

TEST(Gelu, Gradients) {
  SKIP_IF_NO_CUDA();
  check_grads(F::gelu, gelu_ref, 1.5f);
}

TEST(Gelu, LargeInputs) {
  SKIP_IF_NO_CUDA();
  check_large_inputs(F::gelu, {-1000.0f, -50.0f, 0.0f, 50.0f, 1000.0f},
                     {0.0f, 0.0f, 0.0f, 50.0f, 1000.0f});
}

// ---- All activations -------------------------------------------------------------------

TEST(Functional, RejectsCpuTensors) {
  SKIP_IF_NO_CUDA();
  const Tensor cpu = fg::ones({2, 3});
  EXPECT_THROW(F::silu(cpu), fg::Error);
  EXPECT_THROW(F::leaky_relu(cpu), fg::Error);
  EXPECT_THROW(F::tanh(cpu), fg::Error);
  EXPECT_THROW(F::softmax(cpu, 1), fg::Error);
  EXPECT_THROW(F::gelu(cpu), fg::Error);
}

}  // namespace
