#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"

// matmul and softmax (CPU reference).
namespace {

using flashgrad::DType;
using flashgrad::Shape;
using flashgrad::Tensor;

Tensor random_tensor(const Shape& shape, std::uint64_t seed, DType dtype = DType::Float32) {
  flashgrad::Generator gen(seed);
  return flashgrad::randn(shape, {.dtype = dtype}, &gen);
}

// Naive row-major matmul of float64 matrices.
std::vector<double> naive_matmul(const std::vector<double>& a, const std::vector<double>& b,
                                 std::size_t m, std::size_t k, std::size_t n) {
  std::vector<double> out(m * n, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t p = 0; p < k; ++p) out[i * n + j] += a[i * k + p] * b[p * n + j];
    }
  }
  return out;
}

class MatmulSizes
    : public ::testing::TestWithParam<std::tuple<std::int64_t, std::int64_t, std::int64_t>> {};

TEST_P(MatmulSizes, MatchesNaiveLoop) {
  const auto [m, k, n] = GetParam();
  const Tensor a = random_tensor({m, k}, 1, DType::Float64);
  const Tensor b = random_tensor({k, n}, 2, DType::Float64);
  const Tensor out = flashgrad::matmul(a, b);
  ASSERT_EQ(out.shape(), (Shape{m, n}));
  const std::vector<double> expected =
      naive_matmul(a.to_vector<double>(), b.to_vector<double>(), static_cast<std::size_t>(m),
                   static_cast<std::size_t>(k), static_cast<std::size_t>(n));
  // Same summation order, but the compiler may fuse multiply-adds differently.
  EXPECT_TRUE(flashgrad::allclose(out, flashgrad::tensor(expected, {m, n}), 1e-12, 1e-12));
}

// Includes empty dims, size 1, and sizes that are not multiples of typical tile sizes.
INSTANTIATE_TEST_SUITE_P(Cpu, MatmulSizes,
                         ::testing::Values(std::make_tuple(1, 1, 1), std::make_tuple(2, 3, 4),
                                           std::make_tuple(17, 33, 9), std::make_tuple(64, 1, 64),
                                           std::make_tuple(0, 3, 4), std::make_tuple(3, 0, 4),
                                           std::make_tuple(3, 4, 0)));

TEST(Matmul, EmptyInnerDimGivesZeros) {
  const Tensor out = flashgrad::matmul(flashgrad::empty({2, 0}), flashgrad::empty({0, 3}));
  EXPECT_EQ(out.to_vector<float>(), std::vector<float>(6, 0.0f));
}

TEST(Matmul, Float32CloseToFloat64) {
  const Tensor a = random_tensor({31, 47}, 3);
  const Tensor b = random_tensor({47, 29}, 4);
  const Tensor ref =
      flashgrad::matmul(a.to(DType::Float64), b.to(DType::Float64)).to(DType::Float32);
  EXPECT_TRUE(flashgrad::allclose(flashgrad::matmul(a, b), ref, 1e-5, 1e-5));
}

TEST(Matmul, NonContiguousInputs) {
  const Tensor a = random_tensor({5, 4}, 5, DType::Float64);
  const Tensor b = random_tensor({6, 5}, 6, DType::Float64);
  // a.t(): [4, 5], b.t(): [5, 6], both transposed views.
  const Tensor out = flashgrad::matmul(a.t(), b.t());
  const Tensor ref = flashgrad::matmul(a.t().contiguous(), b.t().contiguous());
  EXPECT_TRUE(flashgrad::equal(out, ref));
  // Strided slice of a larger matrix.
  const Tensor big = random_tensor({8, 10}, 7, DType::Float64);
  const Tensor s = big.slice(0, 0, 8, 2).slice(1, 1, 10, 3);  // [4, 3]
  EXPECT_TRUE(flashgrad::equal(flashgrad::matmul(s, s.t()),
                               flashgrad::matmul(s.contiguous(), s.t().contiguous())));
}

TEST(Matmul, IdentityAndKnownValues) {
  const Tensor a = flashgrad::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  EXPECT_TRUE(flashgrad::equal(flashgrad::matmul(a, flashgrad::eye(3)), a));
  const Tensor b = flashgrad::tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
  EXPECT_EQ(flashgrad::matmul(a, b).to_vector<float>(), (std::vector<float>{58, 64, 139, 154}));
}

TEST(Matmul, Integers) {
  const Tensor a = flashgrad::tensor({1, 2, 3, 4}, {2, 2});
  EXPECT_EQ(flashgrad::matmul(a, a).to_vector<std::int32_t>(),
            (std::vector<std::int32_t>{7, 10, 15, 22}));
}

TEST(Matmul, BatchedWithBroadcasting) {
  const Tensor a = random_tensor({2, 1, 3, 4}, 8, DType::Float64);
  const Tensor b = random_tensor({5, 4, 2}, 9, DType::Float64);
  const Tensor out = flashgrad::matmul(a, b);
  ASSERT_EQ(out.shape(), (Shape{2, 5, 3, 2}));
  for (std::int64_t i = 0; i < 2; ++i) {
    for (std::int64_t j = 0; j < 5; ++j) {
      const Tensor expected = flashgrad::matmul(a[i][0].contiguous(), b[j].contiguous());
      EXPECT_TRUE(flashgrad::equal(out[i][j].contiguous(), expected)) << i << "," << j;
    }
  }
}

TEST(Matmul, InvalidArgumentsThrow) {
  EXPECT_THROW(flashgrad::matmul(flashgrad::zeros({3}), flashgrad::zeros({3, 2})),
               flashgrad::Error);
  EXPECT_THROW(flashgrad::matmul(flashgrad::zeros({2, 3}), flashgrad::zeros({4, 2})),
               flashgrad::Error);
  EXPECT_THROW(flashgrad::matmul(flashgrad::zeros({2, 2, 3}), flashgrad::zeros({3, 3, 2})),
               flashgrad::Error);  // batch dims 2 vs 3
  EXPECT_THROW(flashgrad::matmul(flashgrad::zeros({2, 2}),
                                 flashgrad::zeros({2, 2}, {.dtype = DType::Float64})),
               flashgrad::Error);
  EXPECT_THROW(flashgrad::matmul(flashgrad::zeros({2, 2}, {.dtype = DType::Bool}),
                                 flashgrad::zeros({2, 2}, {.dtype = DType::Bool})),
               flashgrad::Error);
}

// ---- Softmax ------------------------------------------------------------------------------

TEST(Softmax, MatchesFormula) {
  const Tensor a = random_tensor({4, 7}, 10, DType::Float64);
  const std::vector<double> x = a.to_vector<double>();
  const std::vector<double> got = flashgrad::softmax(a, 1).to_vector<double>();
  for (std::size_t i = 0; i < 4; ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < 7; ++j) sum += std::exp(x[i * 7 + j]);
    for (std::size_t j = 0; j < 7; ++j) {
      EXPECT_NEAR(got[i * 7 + j], std::exp(x[i * 7 + j]) / sum, 1e-14);
    }
  }
}

TEST(Softmax, RowsSumToOneAlongAnyDim) {
  const Tensor a = random_tensor({3, 4, 5}, 11);
  for (std::int64_t dim = -3; dim < 3; ++dim) {
    const Tensor s = flashgrad::softmax(a, dim);
    EXPECT_EQ(s.shape(), a.shape());
    const Tensor sums = flashgrad::sum(s, dim);
    EXPECT_TRUE(flashgrad::allclose(sums, flashgrad::ones_like(sums), 1e-6, 1e-6)) << dim;
  }
}

TEST(Softmax, NonContiguousInputMatchesContiguous) {
  const Tensor a = random_tensor({6, 5}, 12).t();
  EXPECT_TRUE(flashgrad::equal(flashgrad::softmax(a, 1), flashgrad::softmax(a.contiguous(), 1)));
  EXPECT_TRUE(flashgrad::equal(flashgrad::softmax(a, 0), flashgrad::softmax(a.contiguous(), 0)));
}

TEST(Softmax, NumericallyStableForLargeInputs) {
  const Tensor a = flashgrad::tensor({1000.0f, 1001.0f, 1002.0f});
  const std::vector<float> s = flashgrad::softmax(a, 0).to_vector<float>();
  const Tensor ref = flashgrad::softmax(flashgrad::tensor({0.0f, 1.0f, 2.0f}), 0);
  EXPECT_TRUE(flashgrad::allclose(flashgrad::tensor(s), ref));
  const std::vector<float> ls = flashgrad::log_softmax(a, 0).to_vector<float>();
  for (const float v : ls) EXPECT_TRUE(std::isfinite(v));
}

TEST(Softmax, LogSoftmaxIsLogOfSoftmax) {
  const Tensor a = random_tensor({5, 9}, 13, DType::Float64);
  EXPECT_TRUE(flashgrad::allclose(flashgrad::log_softmax(a, -1),
                                  flashgrad::log(flashgrad::softmax(a, -1)), 1e-12, 1e-12));
}

TEST(Softmax, NegativeInfinityGetsZeroProbability) {
  const float inf = std::numeric_limits<float>::infinity();
  const Tensor a = flashgrad::tensor({0.0f, -inf, 0.0f});
  EXPECT_EQ(flashgrad::softmax(a, 0).to_vector<float>(), (std::vector<float>{0.5f, 0.0f, 0.5f}));
}

TEST(Softmax, EdgeCases) {
  EXPECT_EQ(flashgrad::softmax(flashgrad::zeros({3, 0}), 1).shape(), (Shape{3, 0}));
  EXPECT_EQ(flashgrad::softmax(flashgrad::zeros({1}), 0).to_vector<float>(),
            (std::vector<float>{1.0f}));
  EXPECT_THROW(flashgrad::softmax(flashgrad::zeros({2}, {.dtype = DType::Int32}), 0),
               flashgrad::Error);
  EXPECT_THROW(flashgrad::softmax(flashgrad::zeros({2}), 1), flashgrad::Error);
}

// ---- allclose / equal ---------------------------------------------------------------------

TEST(Compare, AllcloseAndEqual) {
  const Tensor a = flashgrad::tensor({1.0f, 2.0f});
  EXPECT_TRUE(flashgrad::allclose(a, a + 1e-7));
  EXPECT_FALSE(flashgrad::allclose(a, a + 1e-3));
  EXPECT_FALSE(flashgrad::allclose(a, flashgrad::tensor({1.0f, 2.0f, 3.0f})));
  EXPECT_TRUE(flashgrad::allclose(a, flashgrad::tensor({1.0, 2.0})));  // dtypes may differ
  const Tensor nan = flashgrad::tensor({std::numeric_limits<float>::quiet_NaN()});
  EXPECT_FALSE(flashgrad::allclose(nan, nan));

  EXPECT_TRUE(flashgrad::equal(a, a.clone()));
  EXPECT_FALSE(flashgrad::equal(a, a + 1e-7));
  EXPECT_FALSE(flashgrad::equal(a, flashgrad::tensor({1.0, 2.0})));  // dtype differs
}

}  // namespace
