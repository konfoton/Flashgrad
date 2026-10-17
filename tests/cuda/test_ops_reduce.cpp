#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
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

struct ReduceCase {
  std::string name;
  Tensor (*op)(const Tensor&, std::int64_t, bool);
  double (*reference)(double, double);
  double init;
  float (*input)(float);
  // Tolerance per reduced element: float32 accumulation drifts with the row length.
  double rtol_per_element;
  double atol_per_element;
};

void PrintTo(const ReduceCase& c, std::ostream* os) { *os << c.name; }

class ReduceOps : public ::testing::TestWithParam<ReduceCase> {
 protected:
  // Reduces `a` along `dim` and compares with the reference (keepdim and not).
  void check(const Tensor& a, std::int64_t dim) const {
    const ReduceCase& c = GetParam();
    const std::int64_t d = dim < 0 ? dim + a.dim() : dim;
    const std::vector<float> expected = ref::reduce(a.to_vector(), a.shape(), d, c.reference,
                                                    c.init);
    const auto n = static_cast<double>(a.size(d));
    const double rtol = c.rtol_per_element * n, atol = c.atol_per_element * n;

    Shape kept = a.shape();
    kept[d] = 1;
    const Tensor out_keep = c.op(a, dim, true);
    EXPECT_EQ(out_keep.shape(), kept);
    EXPECT_TRUE(ref::close(out_keep.to_vector(), expected, rtol, atol))
        << c.name << " over dim " << dim << " of " << fg::describe(a);

    Shape squeezed = a.shape();
    squeezed.erase(squeezed.begin() + d);
    const Tensor out = c.op(a, dim, false);
    EXPECT_EQ(out.shape(), squeezed);
    EXPECT_TRUE(ref::close(out.to_vector(), expected, rtol, atol));
  }
};

TEST_P(ReduceOps, EveryDimension) {
  SKIP_IF_NO_CUDA();
  const Tensor a = ref::random_cuda({3, 4, 5}, 1, GetParam().input);
  for (const std::int64_t dim : {0, 1, 2, -1}) check(a, dim);
}

TEST_P(ReduceOps, RowLengthsAroundWarpAndBlockSizes) {
  SKIP_IF_NO_CUDA();
  std::uint64_t seed = 2;
  for (const std::int64_t n : {1, 2, 31, 32, 33, 255, 257, 5000}) {
    check(ref::random_cuda({3, n}, seed++, GetParam().input), 1);  // unit-stride rows
    check(ref::random_cuda({n, 3}, seed++, GetParam().input), 0);  // strided rows
  }
}

TEST_P(ReduceOps, NonContiguousInput) {
  SKIP_IF_NO_CUDA();
  const Tensor a =
      ref::random_cuda({6, 7, 40}, 20, GetParam().input).permute({2, 0, 1}).slice(0, 1, 40, 3);
  ASSERT_FALSE(a.is_contiguous());
  for (const std::int64_t dim : {0, 1, 2}) check(a, dim);
  check(ref::random_cuda({1, 9}, 21, GetParam().input).expand({4, 9}), 0);  // stride 0 rows
}

TEST_P(ReduceOps, ManyOutputs) {
  SKIP_IF_NO_CUDA();
  check(ref::random_cuda({70000, 3}, 22, GetParam().input), 1);  // one block per output
}

TEST_P(ReduceOps, EmptyOutput) {
  SKIP_IF_NO_CUDA();
  check(ref::random_cuda({0, 5}, 23, GetParam().input), 1);
}

float identity(float v) { return v; }
// Factors near 1 keep long products away from overflow and underflow.
float near_one(float v) { return 1.0f + 0.05f * v; }

INSTANTIATE_TEST_SUITE_P(
    Ops, ReduceOps,
    ::testing::Values(
        ReduceCase{"sum", fg::sum, [](double x, double y) { return x + y; }, 0.0, identity, 2e-7,
                   2e-7},
        ReduceCase{"prod", fg::prod, [](double x, double y) { return x * y; }, 1.0, near_one,
                   2e-7, 0},
        ReduceCase{"max", fg::max, ref::nan_max, -std::numeric_limits<double>::infinity(),
                   identity, 0, 0}),
    [](const auto& test) { return test.param.name; });

TEST(ReduceOps, EmptyDimensionGivesIdentity) {
  SKIP_IF_NO_CUDA();
  const Tensor a = fg::empty({3, 0}, {.device = fg::Device::cuda()});
  EXPECT_TRUE(ref::close(fg::sum(a, 1).to_vector(), {0.0f, 0.0f, 0.0f}));
  EXPECT_TRUE(ref::close(fg::prod(a, 1).to_vector(), {1.0f, 1.0f, 1.0f}));
  EXPECT_THROW(fg::max(a, 1), fg::Error);
}

TEST(ReduceOps, MaxPropagatesNan) {
  SKIP_IF_NO_CUDA();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  // NaN in different positions, including a lane past the first warp.
  std::vector<float> values(2 * 100, 1.0f);
  values[0] = nan;
  values[100 + 70] = nan;
  const Tensor a = fg::tensor(values, {2, 100}).cuda();
  EXPECT_TRUE(ref::close(fg::max(a, 1).to_vector(), {nan, nan}));
}

TEST(ReduceOps, RejectsInvalidDims) {
  SKIP_IF_NO_CUDA();
  const Tensor a = fg::ones({2, 3}, {.device = fg::Device::cuda()});
  EXPECT_THROW(fg::sum(a, 2), fg::Error);
  EXPECT_THROW(fg::sum(a, -3), fg::Error);
  EXPECT_THROW(fg::sum(fg::ones({}, {.device = fg::Device::cuda()}), 0), fg::Error);
}

}  // namespace
