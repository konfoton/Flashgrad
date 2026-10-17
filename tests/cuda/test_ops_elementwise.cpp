#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "reference.h"

namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
using fg::Shape;
using fg::Tensor;

using Transform = std::function<float(float)>;

float identity(float v) { return v; }

// Inputs covering the edge cases: 0-dim, empty, single element, sizes that aren't multiples
// of the block size, a permuted + sliced (non-contiguous) view and an expanded (stride 0) view.
std::vector<Tensor> unary_inputs(const Transform& t) {
  return {
      ref::random_cuda({}, 1, t),
      ref::random_cuda({0}, 2, t),
      ref::random_cuda({1}, 3, t),
      ref::random_cuda({257}, 4, t),
      ref::random_cuda({3, 1000}, 5, t),
      ref::random_cuda({6, 8, 5}, 6, t).permute({2, 0, 1}).slice(1, 1, 6, 2),
      ref::random_cuda({1, 7}, 7, t).expand({4, 7}),
  };
}

struct UnaryCase {
  std::string name;
  Tensor (*op)(const Tensor&);
  double (*reference)(double);
  Transform input;  // keeps inputs inside the op's domain
  double rtol;      // 0 = exact (a single correctly rounded fp32 operation)
  double atol;
};

void PrintTo(const UnaryCase& c, std::ostream* os) { *os << c.name; }

class UnaryOps : public ::testing::TestWithParam<UnaryCase> {};

TEST_P(UnaryOps, MatchReference) {
  SKIP_IF_NO_CUDA();
  const UnaryCase& c = GetParam();
  for (const Tensor& a : unary_inputs(c.input)) {
    const Tensor out = c.op(a);
    EXPECT_EQ(out.shape(), a.shape());
    EXPECT_TRUE(out.device().is_cuda());
    EXPECT_TRUE(out.is_contiguous());
    EXPECT_TRUE(ref::close(out.to_vector(), ref::unary(a.to_vector(), c.reference), c.rtol,
                           c.atol))
        << c.name << " on " << fg::describe(a);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Ops, UnaryOps,
    ::testing::Values(
        UnaryCase{"neg", fg::neg, [](double x) { return -x; }, identity, 0, 0},
        UnaryCase{"relu", fg::relu, [](double x) { return x > 0 ? x : 0.0; }, identity, 0, 0},
        UnaryCase{"exp", fg::exp, [](double x) { return std::exp(x); }, identity, 1e-6, 0},
        UnaryCase{"log", fg::log, [](double x) { return std::log(x); },
                  [](float v) { return std::abs(v) + 0.01f; }, 1e-6, 1e-6},
        UnaryCase{"sigmoid", fg::sigmoid, [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
                  [](float v) { return 10.0f * v; }, 1e-6, 1e-7},
        UnaryCase{"reciprocal", fg::reciprocal, [](double x) { return 1.0 / x; },
                  [](float v) { return v >= 0.0f ? v + 0.01f : v - 0.01f; }, 0, 0}),
    [](const auto& test) { return test.param.name; });

TEST(UnaryOps, SigmoidSaturatesWithoutOverflow) {
  SKIP_IF_NO_CUDA();
  const Tensor out = fg::sigmoid(fg::tensor({-200.0f, -100.0f, 0.0f, 100.0f, 200.0f}).cuda());
  EXPECT_TRUE(ref::close(out.to_vector(), {0.0f, 0.0f, 0.5f, 1.0f, 1.0f}, 0, 1e-30));
}

// ---- Binary ---------------------------------------------------------------------------

struct BinaryCase {
  std::string name;
  Tensor (*op)(const Tensor&, const Tensor&);
  double (*reference)(double, double);
  Transform input;
};

void PrintTo(const BinaryCase& c, std::ostream* os) { *os << c.name; }

// Rounded to multiples of 0.5 so the comparisons see plenty of ties.
float coarse(float v) { return std::round(2.0f * v) / 2.0f; }

class BinaryOps : public ::testing::TestWithParam<BinaryCase> {};

TEST_P(BinaryOps, MatchReferenceWithBroadcasting) {
  SKIP_IF_NO_CUDA();
  const BinaryCase& c = GetParam();
  const std::vector<std::pair<Shape, Shape>> shapes = {
      {{257}, {257}}, {{4, 1, 5}, {3, 1}}, {{}, {2, 3}}, {{2, 3}, {}},
      {{0, 3}, {3}},  {{1}, {5}},          {{1}, {}},    {{2, 1, 3, 1, 4}, {5, 1}},
  };
  std::uint64_t seed = 10;
  for (const auto& [as, bs] : shapes) {
    const Tensor a = ref::random_cuda(as, seed++, c.input);
    const Tensor b = ref::random_cuda(bs, seed++, c.input);
    const Tensor out = c.op(a, b);
    const Shape os = fg::broadcast_shapes(as, bs);
    EXPECT_EQ(out.shape(), os);
    // All ops here are single correctly rounded fp32 operations, so the match is exact.
    EXPECT_TRUE(ref::close(out.to_vector(),
                           ref::binary(a.to_vector(), as, b.to_vector(), bs, os, c.reference)))
        << c.name << " with shapes " << fg::to_string(as) << " and " << fg::to_string(bs);
  }
}

TEST_P(BinaryOps, MatchReferenceOnNonContiguousViews) {
  SKIP_IF_NO_CUDA();
  const BinaryCase& c = GetParam();
  const Tensor a = ref::random_cuda({9, 6}, 30, c.input).t();                   // [6, 9]
  const Tensor b = ref::random_cuda({6, 20}, 31, c.input).slice(1, 2, 20, 2);  // [6, 9]
  ASSERT_FALSE(a.is_contiguous());
  ASSERT_FALSE(b.is_contiguous());
  const Tensor out = c.op(a, b);
  EXPECT_TRUE(ref::close(out.to_vector(), ref::binary(a.to_vector(), a.shape(), b.to_vector(),
                                                      b.shape(), out.shape(), c.reference)))
      << c.name;
}

INSTANTIATE_TEST_SUITE_P(
    Ops, BinaryOps,
    ::testing::Values(
        BinaryCase{"add", fg::add, [](double x, double y) { return x + y; }, identity},
        BinaryCase{"mul", fg::mul, [](double x, double y) { return x * y; }, identity},
        BinaryCase{"maximum", fg::maximum, ref::nan_max, coarse},
        BinaryCase{"lt", fg::lt, [](double x, double y) { return x < y ? 1.0 : 0.0; }, coarse},
        BinaryCase{"eq", fg::eq, [](double x, double y) { return x == y ? 1.0 : 0.0; }, coarse}),
    [](const auto& test) { return test.param.name; });

TEST(BinaryOps, MaximumPropagatesNan) {
  SKIP_IF_NO_CUDA();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor a = fg::tensor({nan, 1.0f, nan, 3.0f}).cuda();
  const Tensor b = fg::tensor({1.0f, nan, nan, 2.0f}).cuda();
  EXPECT_TRUE(ref::close(fg::maximum(a, b).to_vector(), {nan, nan, nan, 3.0f}));
}

TEST(BinaryOps, LargeBroadcastTensor) {
  SKIP_IF_NO_CUDA();
  // ~16.8M elements, more than 65535 blocks x 256 threads, through the strided (broadcast) path.
  const Shape shape{4097, 4100};
  const Tensor a = ref::random_cuda(shape, 40);
  const Tensor b = ref::random_cuda({4100}, 41);
  const Tensor out = fg::add(a, b);
  EXPECT_TRUE(ref::close(out.to_vector(),
                         ref::binary(a.to_vector(), shape, b.to_vector(), {4100}, shape,
                                     [](double x, double y) { return x + y; })));
}

// ---- Errors ---------------------------------------------------------------------------

TEST(OpsErrors, RejectsCpuTensors) {
  const Tensor cpu = fg::ones({3});
  EXPECT_THROW(fg::exp(cpu), fg::Error);
  EXPECT_THROW(fg::sum(cpu, 0), fg::Error);
  if (fg::cuda::is_available()) {
    EXPECT_THROW(fg::add(cpu.cuda(), cpu), fg::Error);
  }
}

TEST(OpsErrors, RejectsUndefinedTensors) { EXPECT_THROW(fg::neg(Tensor()), fg::Error); }

TEST(OpsErrors, RejectsShapesThatDoNotBroadcast) {
  SKIP_IF_NO_CUDA();
  EXPECT_THROW(fg::add(fg::ones({2, 3}, {.device = fg::Device::cuda()}),
                       fg::ones({4}, {.device = fg::Device::cuda()})),
               fg::Error);
}

TEST(OpsErrors, RejectsTooManyDimensions) {
  SKIP_IF_NO_CUDA();
  const Tensor t = fg::ones(Shape(9, 1), {.device = fg::Device::cuda()});
  EXPECT_THROW(fg::neg(t), fg::Error);
  EXPECT_NO_THROW(fg::neg(t.squeeze(0)));  // 8 dimensions is the limit
}

}  // namespace
