#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"

namespace {

using flashgrad::DType;
using flashgrad::Shape;
using flashgrad::Tensor;

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

Tensor random_tensor(const Shape& shape, std::uint64_t seed) {
  flashgrad::Generator gen(seed);
  return flashgrad::randn(shape, {}, &gen);
}

TEST(Reduce, SumAll) {
  const Tensor a = flashgrad::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  const Tensor s = flashgrad::sum(a);
  EXPECT_EQ(s.dim(), 0);
  EXPECT_EQ(s.dtype(), DType::Float32);
  EXPECT_EQ(s.item(), 21.0f);
  EXPECT_EQ(flashgrad::sum(a.t()).item(), 21.0f);
  EXPECT_EQ(flashgrad::sum(flashgrad::full({}, 4.0)).item(), 4.0f);
}

TEST(Reduce, SumAlongDim) {
  const Tensor a = flashgrad::tensor({1, 2, 3, 4, 5, 6}, {2, 3});
  EXPECT_EQ(flashgrad::sum(a, 0).to_vector(), (std::vector<float>{5, 7, 9}));
  EXPECT_EQ(flashgrad::sum(a, 1).to_vector(), (std::vector<float>{6, 15}));
  EXPECT_EQ(flashgrad::sum(a, -1).shape(), (Shape{2}));
  EXPECT_EQ(flashgrad::sum(a, 0, true).shape(), (Shape{1, 3}));
  EXPECT_EQ(flashgrad::sum(a, 1, true).shape(), (Shape{2, 1}));
  EXPECT_THROW(flashgrad::sum(a, 2), flashgrad::Error);
}

TEST(Reduce, SumMatchesLoopOn3DNonContiguous) {
  // [3, 4, 5] tensor viewed through a permute; reduce each dim and compare with a loop
  // accumulated in double.
  const Tensor base = random_tensor({5, 3, 4}, 11);
  const Tensor a = base.permute({1, 2, 0});  // [3, 4, 5]
  const std::vector<float> x = a.contiguous().to_vector();
  auto at = [&](std::size_t i, std::size_t j, std::size_t k) { return x[(i * 4 + j) * 5 + k]; };
  const std::size_t sizes[3] = {3, 4, 5};
  for (std::int64_t dim = 0; dim < 3; ++dim) {
    const std::vector<float> got = flashgrad::sum(a, dim).to_vector();
    std::vector<float> expected;
    for (std::size_t i = 0; i < (dim == 0 ? 1 : sizes[0]); ++i) {
      for (std::size_t j = 0; j < (dim == 1 ? 1 : sizes[1]); ++j) {
        for (std::size_t k = 0; k < (dim == 2 ? 1 : sizes[2]); ++k) {
          double acc = 0.0;
          for (std::size_t r = 0; r < sizes[dim]; ++r) {
            acc += at(dim == 0 ? r : i, dim == 1 ? r : j, dim == 2 ? r : k);
          }
          expected.push_back(static_cast<float>(acc));
        }
      }
    }
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t n = 0; n < got.size(); ++n) {
      EXPECT_EQ(got[n], expected[n]) << "dim " << dim << " index " << n;
    }
  }
}

TEST(Reduce, SumOfEmpty) {
  EXPECT_EQ(flashgrad::sum(flashgrad::zeros({0})).item(), 0.0f);
  EXPECT_EQ(flashgrad::sum(flashgrad::zeros({2, 0}), 1).to_vector(),
            (std::vector<float>{0, 0}));
  EXPECT_EQ(flashgrad::sum(flashgrad::zeros({2, 0}), 0).shape(), (Shape{0}));
}

TEST(Reduce, SumAccumulatesInDouble) {
  // Accumulating in float32 would give 2^24: 2^24 + 1 rounds back to 2^24.
  const Tensor b = flashgrad::tensor({16777216.0f, 1.0f, 1.0f});
  EXPECT_EQ(flashgrad::sum(b).item(), 16777218.0f);
}

TEST(Reduce, Mean) {
  const Tensor a = flashgrad::tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  EXPECT_EQ(flashgrad::mean(a).item(), 2.5);
  EXPECT_EQ(flashgrad::mean(a, 0).to_vector(), (std::vector<float>{2, 3}));
  EXPECT_EQ(flashgrad::mean(a, 1, true).shape(), (Shape{2, 1}));
  EXPECT_TRUE(std::isnan(flashgrad::mean(flashgrad::zeros({0})).item()));
}

TEST(Reduce, MaxMin) {
  const Tensor a = flashgrad::tensor({3, -1, 7, 2, 9, -5}, {2, 3});
  EXPECT_EQ(flashgrad::max(a).item(), 9);
  EXPECT_EQ(flashgrad::min(a).item(), -5);
  EXPECT_EQ(flashgrad::max(a, 0).to_vector(), (std::vector<float>{3, 9, 7}));
  EXPECT_EQ(flashgrad::min(a, 1).to_vector(), (std::vector<float>{-1, -5}));
  EXPECT_EQ(flashgrad::max(a, 1, true).shape(), (Shape{2, 1}));
}

TEST(Reduce, MaxMinPropagateNaN) {
  const Tensor a = flashgrad::tensor({1.0f, kNaN, 3.0f, 0.0f, 2.0f, -1.0f}, {2, 3});
  EXPECT_TRUE(std::isnan(flashgrad::max(a).item()));
  EXPECT_TRUE(std::isnan(flashgrad::min(a).item()));
  const std::vector<float> row_max = flashgrad::max(a, 1).to_vector();
  EXPECT_TRUE(std::isnan(row_max[0]));
  EXPECT_EQ(row_max[1], 2.0f);
}

TEST(Reduce, MaxOfEmptyThrows) {
  EXPECT_THROW(flashgrad::max(flashgrad::zeros({0})), flashgrad::Error);
  EXPECT_THROW(flashgrad::min(flashgrad::zeros({2, 0}), 1), flashgrad::Error);
  // Nothing to reduce into: fine.
  EXPECT_EQ(flashgrad::max(flashgrad::zeros({0, 3}), 1).shape(), (Shape{0}));
}

TEST(Reduce, ArgmaxArgmin) {
  const Tensor a = flashgrad::tensor({3.0f, -1.0f, 7.0f, 2.0f, 9.0f, -5.0f}, {2, 3});
  const Tensor i = flashgrad::argmax(a);
  EXPECT_EQ(i.dtype(), DType::Float32);  // indices are returned as floats
  EXPECT_EQ(i.item(), 4);  // flattened index
  EXPECT_EQ(flashgrad::argmin(a).item(), 5);
  EXPECT_EQ(flashgrad::argmax(a, 0).to_vector(),
            (std::vector<float>{0, 1, 0}));
  EXPECT_EQ(flashgrad::argmax(a, 1).to_vector(), (std::vector<float>{2, 1}));
  EXPECT_EQ(flashgrad::argmin(a, -1, true).shape(), (Shape{2, 1}));
  // Flattened index follows the logical (row-major) order, not memory order.
  EXPECT_EQ(flashgrad::argmax(a.t()).item(), 3);
}

TEST(Reduce, ArgmaxTiesAndNaN) {
  EXPECT_EQ(flashgrad::argmax(flashgrad::tensor({1, 5, 5, 2})).item(), 1);
  EXPECT_EQ(flashgrad::argmin(flashgrad::tensor({3, 1, 1})).item(), 1);
  EXPECT_EQ(flashgrad::argmax(flashgrad::tensor({1.0f, kNaN, 9.0f, kNaN})).item(), 1);
  EXPECT_EQ(flashgrad::argmin(flashgrad::tensor({1.0f, kNaN, -9.0f})).item(), 1);
  EXPECT_THROW(flashgrad::argmax(flashgrad::zeros({0})), flashgrad::Error);
}

TEST(Reduce, ArgmaxIndexIsExactForLargeDims) {
  // Indices are floats; they are exact for any dim below 2^24 elements.
  const Tensor a = flashgrad::zeros({100001});
  a.slice(0, 100000).fill_(1.0);
  EXPECT_EQ(flashgrad::argmax(a).item(), 100000.0f);
}

}  // namespace
