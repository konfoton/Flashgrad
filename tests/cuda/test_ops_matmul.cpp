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
using fg::Tensor;

// Compares matmul(a, b) with the reference. float32 accumulation drifts with K.
void check(const Tensor& a, const Tensor& b) {
  const std::int64_t m = a.size(0), k = a.size(1), n = b.size(1);
  const std::vector<float> expected = ref::matmul(a.to_vector(), b.to_vector(), m, k, n);
  const Tensor out = fg::matmul(a, b);
  EXPECT_EQ(out.shape(), (fg::Shape{m, n}));
  EXPECT_TRUE(out.is_contiguous());
  EXPECT_TRUE(ref::close(out.to_vector(), expected, 1e-5, 2e-6 * static_cast<double>(k)))
      << fg::describe(a) << " @ " << fg::describe(b);
}

TEST(MatmulOps, SizesAroundTiles) {
  SKIP_IF_NO_CUDA();
  std::uint64_t seed = 1;
  // Block tile is 64 x 64 with K steps of 8; float4 paths need K and N multiples of 4.
  for (const std::int64_t m : {1, 7, 64, 65, 130}) {
    for (const std::int64_t k : {1, 3, 4, 8, 9, 64, 100}) {
      for (const std::int64_t n : {1, 5, 8, 63, 64, 129}) {
        check(ref::random_cuda({m, k}, seed), ref::random_cuda({k, n}, seed + 1));
        seed += 2;
      }
    }
  }
}

TEST(MatmulOps, Large) {
  SKIP_IF_NO_CUDA();
  check(ref::random_cuda({300, 1024}, 10), ref::random_cuda({1024, 260}, 11));  // float4 paths
  check(ref::random_cuda({257, 513}, 12), ref::random_cuda({513, 255}, 13));    // scalar paths
}

TEST(MatmulOps, TransposedInputs) {
  SKIP_IF_NO_CUDA();
  const Tensor a = ref::random_cuda({48, 70}, 20);
  const Tensor b = ref::random_cuda({36, 48}, 21);
  ASSERT_FALSE(a.t().is_contiguous());
  check(a.t(), b.t());                    // [70, 48] @ [48, 36]
  check(a.t(), ref::random_cuda({48, 32}, 22));
  check(ref::random_cuda({33, 36}, 23), b);
  check(b, a);
}

TEST(MatmulOps, StridedAndMisalignedInputs) {
  SKIP_IF_NO_CUDA();
  // Offset by one element: rows are unit-stride with stride % 4 == 0 but not 16-byte aligned.
  const Tensor a = ref::random_cuda({40, 69}, 30).slice(1, 1);
  const Tensor b = ref::random_cuda({68, 41}, 31).slice(1, 1);
  check(a, b);
  // Every other column and row.
  const Tensor c = ref::random_cuda({80, 64}, 32).slice(0, 0, 80, 2).slice(1, 0, 64, 2);
  const Tensor d = ref::random_cuda({64, 50}, 33).slice(0, 0, 64, 2);
  check(c, d);
  // Expanded rows and columns (stride 0).
  check(ref::random_cuda({1, 16}, 34).expand({20, 16}),
        ref::random_cuda({16, 1}, 35).expand({16, 12}));
}

TEST(MatmulOps, EmptyDims) {
  SKIP_IF_NO_CUDA();
  const fg::TensorOptions gpu{.device = fg::Device::cuda()};
  EXPECT_EQ(fg::matmul(fg::empty({0, 5}, gpu), fg::ones({5, 3}, gpu)).shape(), (fg::Shape{0, 3}));
  EXPECT_EQ(fg::matmul(fg::ones({4, 5}, gpu), fg::empty({5, 0}, gpu)).shape(), (fg::Shape{4, 0}));
  // K == 0 gives zeros.
  const Tensor z = fg::matmul(fg::empty({70, 0}, gpu), fg::empty({0, 66}, gpu));
  EXPECT_TRUE(ref::close(z.to_vector(), std::vector<float>(70 * 66, 0.0f)));
}

TEST(MatmulOps, RejectsInvalidInputs) {
  SKIP_IF_NO_CUDA();
  const fg::TensorOptions gpu{.device = fg::Device::cuda()};
  EXPECT_THROW(fg::matmul(fg::ones({2, 3}, gpu), fg::ones({4, 2}, gpu)), fg::Error);
  EXPECT_THROW(fg::matmul(fg::ones({3}, gpu), fg::ones({3, 2}, gpu)), fg::Error);
  EXPECT_THROW(fg::matmul(fg::ones({2, 2, 3}, gpu), fg::ones({3, 2}, gpu)), fg::Error);
  EXPECT_THROW(fg::matmul(fg::ones({2, 3}), fg::ones({3, 2})), fg::Error);  // CPU tensors
  EXPECT_THROW(fg::matmul(fg::ones({2, 3}, gpu), fg::ones({3, 2})), fg::Error);
}

}  // namespace
