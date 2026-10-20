#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "flashgrad/core/ops.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "gradcheck.h"
#include "reference.h"

// Gradients of the elementwise ops, checked against finite differences of the references.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
using fg::Shape;
using fg::Tensor;
using gradcheck::Values;

using Transform = std::function<float(float)>;

float identity(float v) { return v; }
// Keeps values at least 0.5 away from zero, for division and logarithms.
float away_from_zero(float v) { return v >= 0.0f ? v + 0.5f : v - 0.5f; }

// ---- Unary ----------------------------------------------------------------------------

struct UnaryGradCase {
  std::string name;
  Tensor (*op)(const Tensor&);
  double (*reference)(double);
  Transform input = identity;  // keeps inputs inside the op's domain and away from kinks
};

void PrintTo(const UnaryGradCase& c, std::ostream* os) { *os << c.name; }

class UnaryGrads : public ::testing::TestWithParam<UnaryGradCase> {};

TEST_P(UnaryGrads, MatchFiniteDifferences) {
  SKIP_IF_NO_CUDA();
  const UnaryGradCase& c = GetParam();
  const std::vector<Tensor> inputs = {
      ref::random_cuda({}, 1, c.input),
      ref::random_cuda({7}, 2, c.input),
      ref::random_cuda({3, 4}, 3, c.input),
      ref::random_cuda({4, 6}, 4, c.input).t().slice(0, 1, 6, 2),  // non-contiguous
      ref::random_cuda({1, 4}, 5, c.input).expand({3, 4}),          // stride 0
      ref::random_cuda({0, 2}, 6, c.input),
  };
  const auto reference = [&](const std::vector<Values>& in) {
    return ref::unary(in[0], c.reference);
  };
  const auto op = [&](const std::vector<Tensor>& in) { return c.op(in[0]); };
  for (const Tensor& a : inputs) {
    EXPECT_TRUE(gradcheck::check(op, reference, {a})) << c.name << " of " << fg::describe(a);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Ops, UnaryGrads,
    ::testing::Values(UnaryGradCase{"neg", fg::neg, [](double x) { return -x; }},
                      UnaryGradCase{"relu", fg::relu, [](double x) { return x > 0 ? x : 0.0; },
                                    away_from_zero},
                      UnaryGradCase{"exp", fg::exp, [](double x) { return std::exp(x); }},
                      UnaryGradCase{"log", fg::log, [](double x) { return std::log(x); },
                                    [](float v) { return std::abs(v) + 0.5f; }},
                      UnaryGradCase{"sigmoid", fg::sigmoid,
                                    [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
                                    [](float v) { return 3.0f * v; }},
                      UnaryGradCase{"reciprocal", fg::reciprocal, [](double x) { return 1.0 / x; },
                                    away_from_zero}),
    [](const auto& test) { return test.param.name; });

TEST(UnaryGrads, ReluGradientIsZeroAtZero) {
  SKIP_IF_NO_CUDA();
  const Tensor x = fg::tensor({-1.0f, 0.0f, 2.0f}).cuda().requires_grad_();
  fg::relu(x).backward(fg::ones({3}, {.device = fg::Device::cuda()}));
  EXPECT_EQ(x.grad().to_vector(), (std::vector<float>{0, 0, 1}));
}

// ---- Binary ---------------------------------------------------------------------------

struct BinaryGradCase {
  std::string name;
  Tensor (*op)(const Tensor&, const Tensor&);
  double (*reference)(double, double);
  Transform input = identity;  // keeps inputs inside the op's domain and away from kinks
};

void PrintTo(const BinaryGradCase& c, std::ostream* os) { *os << c.name; }

class BinaryGrads : public ::testing::TestWithParam<BinaryGradCase> {
 protected:
  void check(const Tensor& a, const Tensor& b, const gradcheck::Options& options = {}) const {
    const BinaryGradCase& c = GetParam();
    const Shape as = a.shape(), bs = b.shape();
    const Shape os = fg::broadcast_shapes(as, bs);
    const auto reference = [&](const std::vector<Values>& in) {
      return ref::binary(in[0], as, in[1], bs, os, c.reference);
    };
    const auto op = [&](const std::vector<Tensor>& in) { return c.op(in[0], in[1]); };
    EXPECT_TRUE(gradcheck::check(op, reference, {a, b}, options))
        << c.name << " of " << fg::describe(a) << " and " << fg::describe(b);
  }
};

TEST_P(BinaryGrads, SameShape) {
  SKIP_IF_NO_CUDA();
  const Transform& t = GetParam().input;
  check(ref::random_cuda({3, 4}, 1, t), ref::random_cuda({3, 4}, 2, t));
  check(ref::random_cuda({}, 3, t), ref::random_cuda({}, 4, t));
}

TEST_P(BinaryGrads, Broadcasting) {
  SKIP_IF_NO_CUDA();
  const Transform& t = GetParam().input;
  const std::vector<std::pair<Shape, Shape>> shapes = {
      {{2, 3, 4}, {4}}, {{3, 1}, {1, 4}}, {{}, {2, 3}}, {{2, 3}, {}}, {{5, 1, 2}, {3, 1}}};
  std::uint64_t seed = 10;
  for (const auto& [as, bs] : shapes) {
    check(ref::random_cuda(as, seed, t), ref::random_cuda(bs, seed + 1, t));
    seed += 2;
  }
}

TEST_P(BinaryGrads, NonContiguousInputs) {
  SKIP_IF_NO_CUDA();
  const Transform& t = GetParam().input;
  check(ref::random_cuda({4, 3}, 20, t).t(), ref::random_cuda({3, 8}, 21, t).slice(1, 0, 8, 2));
}

TEST_P(BinaryGrads, OnlyOneInputRequiresGrad) {
  SKIP_IF_NO_CUDA();
  const Transform& t = GetParam().input;
  const Tensor a = ref::random_cuda({2, 3}, 30, t), b = ref::random_cuda({3}, 31, t);
  check(a, b, {.requires_grad = {true, false}});
  check(a, b, {.requires_grad = {false, true}});
}

TEST_P(BinaryGrads, EmptyInputs) {
  SKIP_IF_NO_CUDA();
  const Transform& t = GetParam().input;
  check(ref::random_cuda({0, 3}, 40, t), ref::random_cuda({3}, 41, t));
}

INSTANTIATE_TEST_SUITE_P(
    Ops, BinaryGrads,
    ::testing::Values(
        BinaryGradCase{"add", fg::add, [](double x, double y) { return x + y; }},
        BinaryGradCase{"sub", fg::sub, [](double x, double y) { return x - y; }},
        BinaryGradCase{"mul", fg::mul, [](double x, double y) { return x * y; }},
        BinaryGradCase{"div", fg::div, [](double x, double y) { return x / y; }, away_from_zero},
        BinaryGradCase{"maximum", fg::maximum, ref::nan_max}),
    [](const auto& test) { return test.param.name; });

TEST(BinaryGrads, MaximumSplitsTiesEvenly) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  const Tensor a = fg::tensor({1, 5, 3}, gpu).requires_grad_();
  const Tensor b = fg::tensor({2, 5, 1}, gpu).requires_grad_();
  fg::maximum(a, b).backward(fg::ones({3}, {.device = gpu}));
  EXPECT_EQ(a.grad().to_vector(), (std::vector<float>{0, 0.5f, 1}));
  EXPECT_EQ(b.grad().to_vector(), (std::vector<float>{1, 0.5f, 0}));

  const Tensor x = fg::tensor({-1, 2}, gpu).requires_grad_();
  fg::maximum(x, x).backward(fg::ones({2}, {.device = gpu}));  // both halves reach x
  EXPECT_EQ(x.grad().to_vector(), (std::vector<float>{1, 1}));
}

}  // namespace
