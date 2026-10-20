#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "flashgrad/core/ops.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "gradcheck.h"
#include "reference.h"

// Gradients of the reductions, checked against finite differences of the references.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
using fg::Shape;
using fg::Tensor;
using gradcheck::Values;

double add(double x, double y) { return x + y; }

// Inputs for the reductions over a dimension: contiguous, permuted and sliced, expanded.
std::vector<Tensor> inputs() {
  return {
      ref::random_cuda({3, 4, 5}, 1),
      ref::random_cuda({5, 6, 4}, 2).permute({2, 0, 1}).slice(1, 1, 5, 2),  // [4, 2, 6]
      ref::random_cuda({1, 4}, 3).expand({3, 4}),
  };
}

TEST(ReduceGrads, SumAlongEachDimension) {
  SKIP_IF_NO_CUDA();
  for (const Tensor& a : inputs()) {
    for (std::int64_t d = 0; d < a.dim(); ++d) {
      for (const bool keepdim : {false, true}) {
        const auto op = [&](const std::vector<Tensor>& in) { return fg::sum(in[0], d, keepdim); };
        const auto reference = [&](const std::vector<Values>& in) {
          return ref::reduce(in[0], a.shape(), d, add, 0.0);
        };
        EXPECT_TRUE(gradcheck::check(op, reference, {a}))
            << "dim " << d << " keepdim " << keepdim << " of " << fg::describe(a);
      }
    }
  }
}

TEST(ReduceGrads, SumOverAnEmptyDimension) {
  SKIP_IF_NO_CUDA();
  const Tensor a = fg::empty({3, 0}, {.device = fg::Device::cuda()}).requires_grad_();
  fg::sum(a, 1).backward(fg::ones({3}, {.device = fg::Device::cuda()}));
  EXPECT_EQ(a.grad().shape(), (Shape{3, 0}));
}

}  // namespace
