#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"

// Tests for the CPU reference implementation of elementwise ops. Expected values are
// computed independently with plain loops over std::vector.
namespace {

using flashgrad::DType;
using flashgrad::Shape;
using flashgrad::Tensor;

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

Tensor random_tensor(const Shape& shape, std::uint64_t seed) {
  flashgrad::Generator gen(seed);
  return flashgrad::randn(shape, {}, &gen);
}

// ---- Binary ops, checked against the scalar C++ operation ---------------------------------

struct BinaryCase {
  std::string name;
  std::function<Tensor(const Tensor&, const Tensor&)> op;
  std::function<float(float, float)> expected;
};

class BinaryOps : public ::testing::TestWithParam<BinaryCase> {};

TEST_P(BinaryOps, MatchesScalarLoop) {
  const BinaryCase& c = GetParam();
  // Empty, single element, and sizes that are not multiples of typical block sizes.
  for (const std::int64_t n : {0, 1, 7, 255, 257, 1000}) {
    const Tensor a = random_tensor({n}, 1);
    const Tensor b = random_tensor({n}, 2) + 3.0;  // keep away from 0 for div
    const std::vector<float> x = a.to_vector<float>();
    const std::vector<float> y = b.to_vector<float>();
    const Tensor out = c.op(a, b);
    ASSERT_EQ(out.shape(), (Shape{n}));
    const std::vector<float> got = out.to_vector<float>();
    for (std::size_t i = 0; i < got.size(); ++i) {
      // A single fp32 operation is exactly rounded, so exact comparison is valid.
      ASSERT_EQ(got[i], c.expected(x[i], y[i])) << c.name << " at " << i << " (n=" << n << ")";
    }
  }
}

TEST_P(BinaryOps, NonContiguousInputs) {
  const BinaryCase& c = GetParam();
  const Tensor a = random_tensor({5, 7}, 3).t();                          // [7, 5], transposed
  const Tensor b = (random_tensor({7, 10}, 4) + 3.0).slice(1, 0, 10, 2);  // [7, 5], strided
  ASSERT_FALSE(a.is_contiguous());
  ASSERT_FALSE(b.is_contiguous());
  const std::vector<float> x = a.contiguous().to_vector<float>();
  const std::vector<float> y = b.contiguous().to_vector<float>();
  const std::vector<float> got = c.op(a, b).to_vector<float>();
  ASSERT_EQ(got.size(), 35u);
  for (std::size_t i = 0; i < got.size(); ++i) {
    ASSERT_EQ(got[i], c.expected(x[i], y[i])) << c.name << " at " << i;
  }
}

TEST_P(BinaryOps, Broadcasting) {
  const BinaryCase& c = GetParam();
  const Tensor a = random_tensor({4, 1, 3}, 5);  // broadcasts against [2, 1]
  const Tensor b = random_tensor({2, 1}, 6) + 3.0;
  const Tensor out = c.op(a, b);
  ASSERT_EQ(out.shape(), (Shape{4, 2, 3}));
  const std::vector<float> x = a.to_vector<float>();
  const std::vector<float> y = b.to_vector<float>();
  const std::vector<float> got = out.to_vector<float>();
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      for (std::size_t k = 0; k < 3; ++k) {
        ASSERT_EQ(got[(i * 2 + j) * 3 + k], c.expected(x[i * 3 + k], y[j]))
            << c.name << " at " << i << "," << j << "," << k;
      }
    }
  }
}

TEST_P(BinaryOps, ScalarTensorOperand) {
  const BinaryCase& c = GetParam();
  const Tensor a = random_tensor({6}, 7);
  const Tensor s = flashgrad::full({}, 2.5);
  const std::vector<float> x = a.to_vector<float>();
  const std::vector<float> lhs = c.op(a, s).to_vector<float>();
  for (std::size_t i = 0; i < x.size(); ++i) EXPECT_EQ(lhs[i], c.expected(x[i], 2.5f));
  EXPECT_EQ(c.op(s, s).dim(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    Cpu, BinaryOps,
    ::testing::Values(
        BinaryCase{"add", [](const Tensor& a, const Tensor& b) { return a + b; },
                   [](float x, float y) { return x + y; }},
        BinaryCase{"sub", [](const Tensor& a, const Tensor& b) { return a - b; },
                   [](float x, float y) { return x - y; }},
        BinaryCase{"mul", [](const Tensor& a, const Tensor& b) { return a * b; },
                   [](float x, float y) { return x * y; }},
        BinaryCase{"div", [](const Tensor& a, const Tensor& b) { return a / b; },
                   [](float x, float y) { return x / y; }},
        BinaryCase{"maximum",
                   [](const Tensor& a, const Tensor& b) { return flashgrad::maximum(a, b); },
                   [](float x, float y) { return std::max(x, y); }},
        BinaryCase{"minimum",
                   [](const Tensor& a, const Tensor& b) { return flashgrad::minimum(a, b); },
                   [](float x, float y) { return std::min(x, y); }},
        BinaryCase{
            "pow",
            [](const Tensor& a, const Tensor& b) { return flashgrad::pow(flashgrad::abs(a), b); },
            [](float x, float y) { return std::pow(std::abs(x), y); }}),
    [](const auto& info) { return info.param.name; });

// ---- Unary ops ------------------------------------------------------------------------------

struct UnaryCase {
  std::string name;
  std::function<Tensor(const Tensor&)> op;
  std::function<float(float)> expected;
};

class UnaryOps : public ::testing::TestWithParam<UnaryCase> {};

TEST_P(UnaryOps, MatchesScalarLoop) {
  const UnaryCase& c = GetParam();
  for (const std::int64_t n : {0, 1, 257}) {
    const Tensor a = random_tensor({n}, 8);
    const std::vector<float> x = a.to_vector<float>();
    const std::vector<float> got = c.op(a).to_vector<float>();
    ASSERT_EQ(got.size(), x.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      ASSERT_EQ(got[i], c.expected(x[i])) << c.name << " at " << i;
    }
  }
}

TEST_P(UnaryOps, NonContiguousInput) {
  const UnaryCase& c = GetParam();
  const Tensor a = random_tensor({3, 4, 5}, 9).permute({2, 0, 1});
  const std::vector<float> x = a.contiguous().to_vector<float>();
  const Tensor out = c.op(a);
  EXPECT_EQ(out.shape(), (Shape{5, 3, 4}));
  EXPECT_TRUE(out.is_contiguous());
  const std::vector<float> got = out.to_vector<float>();
  for (std::size_t i = 0; i < got.size(); ++i) {
    ASSERT_EQ(got[i], c.expected(x[i])) << c.name << " at " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Cpu, UnaryOps,
    ::testing::Values(
        UnaryCase{"neg", [](const Tensor& a) { return -a; }, [](float x) { return -x; }},
        UnaryCase{"abs", [](const Tensor& a) { return flashgrad::abs(a); },
                  [](float x) { return std::abs(x); }},
        UnaryCase{"relu", [](const Tensor& a) { return flashgrad::relu(a); },
                  [](float x) { return x < 0.0f ? 0.0f : x; }},
        UnaryCase{"exp", [](const Tensor& a) { return flashgrad::exp(a); },
                  [](float x) { return std::exp(x); }},
        UnaryCase{"log", [](const Tensor& a) { return flashgrad::log(flashgrad::abs(a)); },
                  [](float x) { return std::log(std::abs(x)); }},
        UnaryCase{"sqrt", [](const Tensor& a) { return flashgrad::sqrt(flashgrad::abs(a)); },
                  [](float x) { return std::sqrt(std::abs(x)); }},
        UnaryCase{"tanh", [](const Tensor& a) { return flashgrad::tanh(a); },
                  [](float x) { return std::tanh(x); }},
        UnaryCase{"sigmoid", [](const Tensor& a) { return flashgrad::sigmoid(a); },
                  [](float x) { return 1.0f / (1.0f + std::exp(-x)); }}),
    [](const auto& info) { return info.param.name; });

// ---- Dtypes and special values ----------------------------------------------------------

TEST(Elementwise, IntegerArithmetic) {
  const Tensor a = flashgrad::tensor({7, -7, 9, 0});
  const Tensor b = flashgrad::tensor({2, 2, -4, 5});
  EXPECT_EQ((a + b).to_vector<std::int32_t>(), (std::vector<std::int32_t>{9, -5, 5, 5}));
  EXPECT_EQ((a * b).to_vector<std::int32_t>(), (std::vector<std::int32_t>{14, -14, -36, 0}));
  // Integer division truncates toward zero, like C++.
  EXPECT_EQ((a / b).to_vector<std::int32_t>(), (std::vector<std::int32_t>{3, -3, -2, 0}));
  EXPECT_EQ(flashgrad::abs(a).to_vector<std::int32_t>(), (std::vector<std::int32_t>{7, 7, 9, 0}));
  EXPECT_EQ(flashgrad::relu(a).to_vector<std::int32_t>(), (std::vector<std::int32_t>{7, 0, 9, 0}));
}

TEST(Elementwise, IntegerDivisionByZeroThrows) {
  EXPECT_THROW(flashgrad::tensor({1, 2}) / flashgrad::tensor({1, 0}), flashgrad::Error);
}

TEST(Elementwise, Int64AndFloat64) {
  const Tensor a = flashgrad::tensor({std::int64_t{1} << 40, std::int64_t{3}});
  EXPECT_EQ((a + a).to_vector<std::int64_t>(),
            (std::vector<std::int64_t>{std::int64_t{1} << 41, 6}));
  const Tensor d = flashgrad::tensor({0.1, 0.2});
  EXPECT_EQ((d + d).to_vector<double>(), (std::vector<double>{0.1 + 0.1, 0.2 + 0.2}));
}

TEST(Elementwise, FloatSpecialValues) {
  const Tensor a = flashgrad::tensor({1.0f, kNaN, kInf, -kInf, 0.0f});
  const Tensor b = flashgrad::tensor({kNaN, 2.0f, 1.0f, 1.0f, 0.0f});
  const std::vector<float> mx = flashgrad::maximum(a, b).to_vector<float>();
  EXPECT_TRUE(std::isnan(mx[0]));  // NaN propagates from either side
  EXPECT_TRUE(std::isnan(mx[1]));
  EXPECT_EQ(mx[2], kInf);
  EXPECT_EQ(mx[3], 1.0f);
  const std::vector<float> mn = flashgrad::minimum(a, b).to_vector<float>();
  EXPECT_TRUE(std::isnan(mn[0]));
  EXPECT_TRUE(std::isnan(mn[1]));
  EXPECT_EQ(mn[3], -kInf);
  EXPECT_TRUE(std::isnan((a / b).to_vector<float>()[4]));  // 0 / 0
  EXPECT_TRUE(std::isnan(flashgrad::relu(a).to_vector<float>()[1]));
  EXPECT_EQ(flashgrad::sigmoid(a).to_vector<float>()[3], 0.0f);
  EXPECT_EQ(flashgrad::exp(a).to_vector<float>()[3], 0.0f);
}

TEST(Elementwise, ScalarOperators) {
  const Tensor a = flashgrad::tensor({1.0f, 2.0f, 4.0f});
  EXPECT_EQ((a + 1.0).to_vector<float>(), (std::vector<float>{2, 3, 5}));
  EXPECT_EQ((a - 1.0).to_vector<float>(), (std::vector<float>{0, 1, 3}));
  EXPECT_EQ((a * 2.0).to_vector<float>(), (std::vector<float>{2, 4, 8}));
  EXPECT_EQ((a / 2.0).to_vector<float>(), (std::vector<float>{0.5, 1, 2}));
  EXPECT_EQ((1.0 + a).to_vector<float>(), (std::vector<float>{2, 3, 5}));
  EXPECT_EQ((10.0 - a).to_vector<float>(), (std::vector<float>{9, 8, 6}));
  EXPECT_EQ((2.0 * a).to_vector<float>(), (std::vector<float>{2, 4, 8}));
  EXPECT_EQ((4.0 / a).to_vector<float>(), (std::vector<float>{4, 2, 1}));
  EXPECT_EQ(flashgrad::pow(a, 2.0).to_vector<float>(), (std::vector<float>{1, 4, 16}));
  // Scalars take the tensor's dtype.
  EXPECT_EQ((flashgrad::tensor({5, 6}) * 2.7).to_vector<std::int32_t>(),
            (std::vector<std::int32_t>{10, 12}));
}

TEST(Elementwise, Comparisons) {
  const Tensor a = flashgrad::tensor({1, 2, 3});
  const Tensor b = flashgrad::tensor({2, 2, 2});
  EXPECT_EQ(flashgrad::eq(a, b).dtype(), DType::Bool);
  EXPECT_EQ(flashgrad::eq(a, b).to_vector<bool>(), (std::vector<bool>{false, true, false}));
  EXPECT_EQ(flashgrad::ne(a, b).to_vector<bool>(), (std::vector<bool>{true, false, true}));
  EXPECT_EQ(flashgrad::lt(a, b).to_vector<bool>(), (std::vector<bool>{true, false, false}));
  EXPECT_EQ(flashgrad::le(a, b).to_vector<bool>(), (std::vector<bool>{true, true, false}));
  EXPECT_EQ(flashgrad::gt(a, b).to_vector<bool>(), (std::vector<bool>{false, false, true}));
  EXPECT_EQ(flashgrad::ge(a, b).to_vector<bool>(), (std::vector<bool>{false, true, true}));
  // Broadcasting and bool inputs.
  EXPECT_EQ(flashgrad::gt(a.view({3, 1}), a).shape(), (Shape{3, 3}));
  const Tensor t = flashgrad::tensor({true, false});
  EXPECT_EQ(flashgrad::eq(t, t).to_vector<bool>(), (std::vector<bool>{true, true}));
  // NaN compares unequal to everything.
  const Tensor n = flashgrad::tensor({kNaN});
  EXPECT_EQ(flashgrad::eq(n, n).to_vector<bool>(), (std::vector<bool>{false}));
  EXPECT_EQ(flashgrad::ne(n, n).to_vector<bool>(), (std::vector<bool>{true}));
}

TEST(Elementwise, InvalidArgumentsThrow) {
  const Tensor f = flashgrad::zeros({3});
  const Tensor i = flashgrad::zeros({3}, {.dtype = DType::Int32});
  const Tensor b = flashgrad::zeros({3}, {.dtype = DType::Bool});
  EXPECT_THROW(f + i, flashgrad::Error);                      // no implicit promotion
  EXPECT_THROW(f + flashgrad::zeros({4}), flashgrad::Error);  // not broadcastable
  EXPECT_THROW(b + b, flashgrad::Error);                      // no arithmetic on bool
  EXPECT_THROW(flashgrad::exp(i), flashgrad::Error);          // floating only
  EXPECT_THROW(flashgrad::pow(i, i), flashgrad::Error);
  EXPECT_THROW(-b, flashgrad::Error);
  EXPECT_THROW(f + Tensor(), flashgrad::Error);
}

TEST(Elementwise, ErrorMessageHasOpShapeDtypeDevice) {
  try {
    flashgrad::add(flashgrad::zeros({2, 3}), flashgrad::zeros({2, 3}, {.dtype = DType::Int64}));
    FAIL() << "expected an error";
  } catch (const flashgrad::Error& e) {
    const std::string msg = e.message();
    EXPECT_NE(msg.find("add"), std::string::npos) << msg;
    EXPECT_NE(msg.find("[2, 3]"), std::string::npos) << msg;
    EXPECT_NE(msg.find("int64"), std::string::npos) << msg;
    EXPECT_NE(msg.find("cpu"), std::string::npos) << msg;
  }
}

// ---- In-place ---------------------------------------------------------------------------

TEST(Elementwise, InPlaceOps) {
  Tensor a = flashgrad::tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  const void* before = a.data_ptr();
  a += flashgrad::tensor({10.0f, 20.0f});  // broadcasts over rows
  EXPECT_EQ(a.to_vector<float>(), (std::vector<float>{11, 22, 13, 24}));
  a -= 1.0;
  a *= 2.0;
  a /= 4.0;
  EXPECT_EQ(a.to_vector<float>(), (std::vector<float>{5, 10.5, 6, 11.5}));
  EXPECT_EQ(a.data_ptr(), before);  // no reallocation
  a.mul_(a);
  EXPECT_EQ(a.to_vector<float>(), (std::vector<float>{25, 110.25, 36, 132.25}));
}

TEST(Elementwise, InPlaceIntoView) {
  const Tensor a = flashgrad::zeros({3, 3});
  a.select(1, 0).add_(flashgrad::tensor({1.0f, 2.0f, 3.0f}));
  EXPECT_EQ(a.to_vector<float>(), (std::vector<float>{1, 0, 0, 2, 0, 0, 3, 0, 0}));
}

TEST(Elementwise, InPlaceWithOverlappingOperand) {
  // a += a.t() must read the original values of a.t(), not partially updated ones.
  const Tensor a = flashgrad::tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  a.add_(a.t());
  EXPECT_EQ(a.to_vector<float>(), (std::vector<float>{2, 5, 5, 8}));
}

TEST(Elementwise, InPlaceShapeMustNotGrow) {
  const Tensor a = flashgrad::zeros({3});
  EXPECT_THROW(a.add_(flashgrad::zeros({2, 3})), flashgrad::Error);
  EXPECT_THROW(flashgrad::zeros({1}).expand({3}).add_(a), flashgrad::Error);
}

}  // namespace
