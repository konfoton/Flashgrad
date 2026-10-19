#include <gtest/gtest.h>

#include <cstdint>
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

// Compares matmul(a, b) with the reference. float32 accumulation drifts with k, and the
// products of standard normals sum to values of size ~sqrt(k).
void check(const Tensor& a, const Tensor& b) {
  const std::int64_t m = a.size(0), k = a.size(1), n = b.size(1);
  const Tensor out = fg::matmul(a, b);
  EXPECT_EQ(out.shape(), (Shape{m, n}));
  EXPECT_TRUE(out.device().is_cuda());
  EXPECT_TRUE(out.is_contiguous());
  const double tol = 2e-7 * static_cast<double>(k + 1);
  EXPECT_TRUE(ref::close(out.to_vector(), ref::matmul(a.to_vector(), b.to_vector(), m, k, n),
                         tol, tol))
      << fg::describe(a) << " @ " << fg::describe(b);
}

TEST(MatmulOp, SizesAroundTheTileSize) {
  SKIP_IF_NO_CUDA();
  std::uint64_t seed = 1;
  for (const std::int64_t m : {1, 15, 16, 17, 64}) {
    for (const std::int64_t k : {1, 16, 33}) {
      for (const std::int64_t n : {1, 16, 31}) {
        check(ref::random_cuda({m, k}, seed), ref::random_cuda({k, n}, seed + 1));
        seed += 2;
      }
    }
  }
}

TEST(MatmulOp, LargerShapes) {
  SKIP_IF_NO_CUDA();
  check(ref::random_cuda({129, 1000}, 1), ref::random_cuda({1000, 70}, 2));  // long k
  check(ref::random_cuda({3000, 8}, 3), ref::random_cuda({8, 300}, 4));      // many tiles
  check(ref::random_cuda({1, 5000}, 5), ref::random_cuda({5000, 1}, 6));     // dot product
}

TEST(MatmulOp, NonContiguousInputs) {
  SKIP_IF_NO_CUDA();
  const Tensor a_t = ref::random_cuda({37, 20}, 1).t();  // [20, 37], column-major
  const Tensor b_t = ref::random_cuda({18, 37}, 2).t();  // [37, 18]
  ASSERT_FALSE(a_t.is_contiguous());
  check(a_t, b_t);
  const Tensor a_sliced = ref::random_cuda({40, 50}, 3).slice(0, 1, 40, 2).slice(1, 3, 50, 3);
  check(a_sliced, ref::random_cuda({a_sliced.size(1), 9}, 4));
  check(ref::random_cuda({5, 1}, 5).expand({5, 21}), ref::random_cuda({21, 1}, 6).expand({21, 3}));
}

TEST(MatmulOp, EmptyInnerDimensionGivesZeros) {
  SKIP_IF_NO_CUDA();
  const Tensor a = fg::empty({3, 0}, {.device = fg::Device::cuda()});
  const Tensor b = fg::empty({0, 4}, {.device = fg::Device::cuda()});
  EXPECT_TRUE(ref::close(fg::matmul(a, b).to_vector(), std::vector<float>(12, 0.0f)));
}

TEST(MatmulOp, EmptyOutput) {
  SKIP_IF_NO_CUDA();
  check(ref::random_cuda({0, 5}, 1), ref::random_cuda({5, 3}, 2));
  check(ref::random_cuda({4, 5}, 3), ref::random_cuda({5, 0}, 4));
}

TEST(MatmulOp, SmallExactValues) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  const Tensor a = fg::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, gpu);
  const Tensor b = fg::tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}, gpu);
  EXPECT_EQ(fg::matmul(a, b).to_vector(), (std::vector<float>{58.0f, 64.0f, 139.0f, 154.0f}));
}

TEST(MatmulOp, RejectsInvalidInputs) {
  SKIP_IF_NO_CUDA();
  const fg::TensorOptions gpu{.device = fg::Device::cuda()};
  EXPECT_THROW(fg::matmul(fg::ones({2, 3}, gpu), fg::ones({4, 2}, gpu)), fg::Error);
  EXPECT_THROW(fg::matmul(fg::ones({3}, gpu), fg::ones({3, 2}, gpu)), fg::Error);
  EXPECT_THROW(fg::matmul(fg::ones({2, 2, 3}, gpu), fg::ones({3, 2}, gpu)), fg::Error);
  EXPECT_THROW(fg::matmul(fg::ones({2, 3}), fg::ones({3, 2})), fg::Error);  // CPU tensors
  EXPECT_THROW(fg::matmul(fg::ones({2, 3}, gpu), Tensor()), fg::Error);
}

}  // namespace
