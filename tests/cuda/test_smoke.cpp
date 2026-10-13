#include "flashgrad/cuda/smoke.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/cuda/runtime.h"

namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

using flashgrad::cuda::smoke::vector_add;

std::vector<float> random_vector(std::size_t n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(gen);
  return v;
}

class VectorAddSizes : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VectorAddSizes, MatchesCpu) {
  SKIP_IF_NO_CUDA();
  const std::size_t n = GetParam();
  const auto a = random_vector(n, 1);
  const auto b = random_vector(n, 2);

  const auto out = vector_add(a, b);

  ASSERT_EQ(out.size(), n);
  for (std::size_t i = 0; i < n; ++i) {
    // Single fp32 add is exactly rounded on both CPU and GPU.
    ASSERT_EQ(out[i], a[i] + b[i]) << "mismatch at index " << i;
  }
}

// Includes edge cases: empty, single element, non-multiples of the block size,
// and a size large enough to exercise the grid-stride loop (> 65535 * 256).
INSTANTIATE_TEST_SUITE_P(Smoke, VectorAddSizes,
                         ::testing::Values(0, 1, 255, 256, 257, 1'000'003, 20'000'000));

TEST(VectorAdd, SizeMismatchThrows) {
  EXPECT_THROW(vector_add(std::vector<float>(3), std::vector<float>(4)), flashgrad::Error);
}

}  // namespace
