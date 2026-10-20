#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
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
using fg::Shape;
using fg::Tensor;

const fg::TensorOptions kGpu{.device = fg::Device::cuda()};

std::vector<float> filled(const Shape& shape, float value) {
  return std::vector<float>(static_cast<std::size_t>(ref::count(shape)), value);
}

// Fills `view` of `base` (a contiguous CUDA tensor) and checks that exactly the view's
// elements changed.
void check_fill_view(const Tensor& base, const Tensor& view, float value) {
  ASSERT_TRUE(base.is_contiguous());
  ASSERT_TRUE(view.storage().is_same(base.storage()));
  std::vector<float> expected = base.to_vector();
  ref::fill_view(expected, view.shape(), view.strides(), view.offset() - base.offset(), value);
  view.fill_(value);
  EXPECT_TRUE(ref::close(base.to_vector(), expected)) << fg::describe(view);
}

TEST(FillOp, FactoriesFillEveryElement) {
  SKIP_IF_NO_CUDA();
  for (const Shape& shape : {Shape{}, Shape{0}, Shape{1}, Shape{257}, Shape{3, 1000},
                             Shape{2, 0, 3}, Shape{4, 3, 2, 5}}) {
    EXPECT_EQ(fg::zeros(shape, kGpu).to_vector(), filled(shape, 0.0f)) << fg::to_string(shape);
    EXPECT_EQ(fg::ones(shape, kGpu).to_vector(), filled(shape, 1.0f)) << fg::to_string(shape);
    EXPECT_EQ(fg::full(shape, -2.5, kGpu).to_vector(), filled(shape, -2.5f))
        << fg::to_string(shape);
  }
  const Tensor like = ref::random_cuda({7, 9}, 1);
  EXPECT_EQ(fg::zeros_like(like).to_vector(), filled({7, 9}, 0.0f));
  EXPECT_EQ(fg::ones_like(like).to_vector(), filled({7, 9}, 1.0f));
  EXPECT_EQ(fg::full_like(like, 4.0).to_vector(), filled({7, 9}, 4.0f));
  EXPECT_TRUE(fg::ones_like(like).device().is_cuda());
}

TEST(FillOp, LargeTensor) {
  SKIP_IF_NO_CUDA();
  const Shape shape{1 << 22, 3};  // more blocks than one wave of the GPU
  EXPECT_EQ(fg::full(shape, 0.5, kGpu).to_vector(), filled(shape, 0.5f));
}

TEST(FillOp, ValueIsRoundedToFloat) {
  SKIP_IF_NO_CUDA();
  EXPECT_EQ(fg::full({3}, 0.1, kGpu).to_vector(), filled({3}, 0.1f));
}

TEST(FillOp, SpecialValues) {
  SKIP_IF_NO_CUDA();
  const float inf = std::numeric_limits<float>::infinity();
  EXPECT_EQ(fg::full({5}, -inf, kGpu).to_vector(), filled({5}, -inf));
  EXPECT_TRUE(ref::close(fg::full({5}, std::nan(""), kGpu).to_vector(),
                         filled({5}, std::numeric_limits<float>::quiet_NaN())));
  // -0.0 keeps its sign bit.
  EXPECT_TRUE(std::signbit(fg::full({}, -0.0, kGpu).item()));
}

TEST(FillOp, OverwritesExistingValues) {
  SKIP_IF_NO_CUDA();
  const Tensor t = ref::random_cuda({300}, 1);
  t.fill_(3.0);
  EXPECT_EQ(t.to_vector(), filled({300}, 3.0f));
  t.zero_();
  EXPECT_EQ(t.to_vector(), filled({300}, 0.0f));
}

TEST(FillOp, NonContiguousViews) {
  SKIP_IF_NO_CUDA();
  const Tensor m = ref::random_cuda({4, 6}, 1);
  check_fill_view(m, m.slice(1, 0, 3), 1.0f);         // left half of every row
  check_fill_view(m, m.select(1, 2), 2.0f);           // one column
  check_fill_view(m, m.slice(0, 1, 4, 2), 3.0f);      // every other row
  check_fill_view(m, m.t(), 4.0f);                    // transposed: every element
  check_fill_view(m, m.slice(1, 1, 6, 2).t(), 5.0f);  // strided and transposed
  const Tensor c = ref::random_cuda({6, 8, 5}, 2);
  check_fill_view(c, c.permute({2, 0, 1}).slice(1, 1, 6, 2), 6.0f);
  check_fill_view(c, c.select(0, 3).select(1, 4), 7.0f);  // 1-D with stride 5, offset > 0
  check_fill_view(c, c.slice(2, 2, 2), 8.0f);              // empty view: nothing changes
  check_fill_view(c, c.select(0, 5)[7][4], 9.0f);          // 0-dim view, last element
}

TEST(FillOp, ExpandedView) {
  SKIP_IF_NO_CUDA();
  // Every element of the view is the same memory location; filling it is still well defined.
  const Tensor base = ref::random_cuda({3, 1}, 1);
  check_fill_view(base, base.expand({3, 4}), 2.0f);
}

TEST(FillOp, DoesNotWaitForQueuedKernels) {
  SKIP_IF_NO_CUDA();
  // Queue a long matmul, then fill: the host must get control back long before the GPU is
  // done. A host -> device cudaMemcpy would wait for the matmul. (cudaStreamQuery can't tell:
  // such a copy may return while its own transfer is still running.) The clock measures how
  // long the host is blocked, not GPU work.
  using Clock = std::chrono::steady_clock;
  const Tensor a = fg::ones({4096, 4096}, kGpu);
  const Tensor dst = fg::empty({1000}, kGpu);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const Clock::time_point start = Clock::now();
  const Tensor product = fg::matmul(a, a);
  const Clock::time_point queued = Clock::now();
  dst.fill_(1.0);
  const Tensor z = fg::zeros({1000}, kGpu);
  const Tensor o = fg::ones_like(dst);
  const Clock::time_point filled_at = Clock::now();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const Clock::time_point done = Clock::now();
  EXPECT_LT(filled_at - queued, (done - start) / 4)
      << "fills took " << std::chrono::duration<double, std::milli>(filled_at - queued).count()
      << " ms of " << std::chrono::duration<double, std::milli>(done - start).count() << " ms";
  EXPECT_EQ(dst.to_vector(), filled({1000}, 1.0f));
  EXPECT_EQ(z.to_vector(), filled({1000}, 0.0f));
  EXPECT_EQ(o.to_vector(), filled({1000}, 1.0f));
}

TEST(FillOp, DimensionLimitOnlyForNonContiguous) {
  SKIP_IF_NO_CUDA();
  Shape nine_dims(9, 1);
  nine_dims.back() = 3;
  EXPECT_EQ(fg::full(nine_dims, 2.0, kGpu).to_vector(), filled(nine_dims, 2.0f));
  const Tensor strided = fg::zeros(nine_dims, kGpu).slice(8, 0, 3, 2);
  ASSERT_FALSE(strided.is_contiguous());
  EXPECT_THROW(strided.fill_(1.0), fg::Error);
  EXPECT_NO_THROW(strided.squeeze(0).fill_(1.0));  // 8 dimensions is the limit
}

}  // namespace
