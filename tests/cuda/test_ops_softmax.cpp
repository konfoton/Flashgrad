#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
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
using fg::Tensor;

// Compares log_softmax(a, dim) with the reference.
void check(const Tensor& a, std::int64_t dim) {
  const std::int64_t d = dim < 0 ? dim + a.dim() : dim;
  const Tensor out = fg::log_softmax(a, dim);
  EXPECT_EQ(out.shape(), a.shape());
  EXPECT_TRUE(out.device().is_cuda());
  EXPECT_TRUE(out.is_contiguous());
  EXPECT_TRUE(ref::close(out.to_vector(), ref::log_softmax(a.to_vector(), a.shape(), d), 1e-5,
                         1e-5))
      << "log_softmax over dim " << dim << " of " << fg::describe(a);
}

TEST(LogSoftmaxOp, EveryDimension) {
  SKIP_IF_NO_CUDA();
  const Tensor a = ref::random_cuda({3, 4, 5}, 1);
  for (const std::int64_t dim : {0, 1, 2, -1}) check(a, dim);
}

TEST(LogSoftmaxOp, RowLengthsAroundWarpAndBlockSizes) {
  SKIP_IF_NO_CUDA();
  std::uint64_t seed = 2;
  for (const std::int64_t n : {1, 2, 31, 32, 33, 255, 257, 5000}) {
    check(ref::random_cuda({3, n}, seed++), 1);  // unit-stride rows
    check(ref::random_cuda({n, 3}, seed++), 0);  // strided rows
  }
}

TEST(LogSoftmaxOp, NonContiguousInput) {
  SKIP_IF_NO_CUDA();
  const Tensor a = ref::random_cuda({6, 7, 40}, 20).permute({2, 0, 1}).slice(0, 1, 40, 3);
  ASSERT_FALSE(a.is_contiguous());
  for (const std::int64_t dim : {0, 1, 2}) check(a, dim);
  check(ref::random_cuda({1, 9}, 21).expand({4, 9}), 1);  // stride 0 rows
  check(ref::random_cuda({1, 9}, 22).expand({4, 9}), 0);  // stride 0 along dim
}

TEST(LogSoftmaxOp, ManyRows) {
  SKIP_IF_NO_CUDA();
  check(ref::random_cuda({70000, 3}, 23), 1);
}

TEST(LogSoftmaxOp, LargeInputsDontOverflow) {
  SKIP_IF_NO_CUDA();
  // exp(1000) overflows float32; without subtracting the max this would give NaN.
  const Tensor a = fg::tensor({1000.0f, 1001.0f, 1002.0f, -1000.0f, -1001.0f, -1002.0f}, {2, 3},
                              fg::Device::cuda());
  check(a, 1);
  const std::vector<float> out = fg::log_softmax(a, 1).to_vector();
  for (const float v : out) EXPECT_TRUE(std::isfinite(v));
}

TEST(LogSoftmaxOp, ExpOfRowSumsToOne) {
  SKIP_IF_NO_CUDA();
  const Tensor probs = fg::exp(fg::log_softmax(ref::random_cuda({8, 300}, 24) * 5.0f, 1));
  EXPECT_TRUE(ref::close(fg::sum(probs, 1).to_vector(), std::vector<float>(8, 1.0f), 1e-5,
                         1e-5));
}

TEST(LogSoftmaxOp, InfinityAndNan) {
  SKIP_IF_NO_CUDA();
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  // A -inf entry gets probability 0; a NaN anywhere makes its row NaN.
  const Tensor a = fg::tensor({0.0f, -inf, 0.0f, 1.0f, nan, 2.0f}, {2, 3}, fg::Device::cuda());
  const std::vector<float> out = fg::log_softmax(a, 1).to_vector();
  const float log_half = std::log(0.5f);
  EXPECT_TRUE(ref::close(out, {log_half, -inf, log_half, nan, nan, nan}, 1e-6, 1e-6));
}

TEST(LogSoftmaxOp, EmptyInput) {
  SKIP_IF_NO_CUDA();
  check(ref::random_cuda({0, 5}, 25), 1);
  check(ref::random_cuda({4, 0}, 26), 1);
}

TEST(LogSoftmaxOp, RejectsInvalidInputs) {
  SKIP_IF_NO_CUDA();
  const fg::TensorOptions gpu{.device = fg::Device::cuda()};
  EXPECT_THROW(fg::log_softmax(fg::ones({2, 3}, gpu), 2), fg::Error);
  EXPECT_THROW(fg::log_softmax(fg::ones({2, 3}, gpu), -3), fg::Error);
  EXPECT_THROW(fg::log_softmax(fg::ones({}, gpu), 0), fg::Error);
  EXPECT_THROW(fg::log_softmax(fg::ones({2, 3}), 1), fg::Error);  // CPU tensor
}

}  // namespace
