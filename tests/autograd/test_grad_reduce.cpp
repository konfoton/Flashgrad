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

TEST(ReduceGrads, SumAndMeanOfAllElements) {
  SKIP_IF_NO_CUDA();
  for (const Tensor& a : {ref::random_cuda({}, 10), ref::random_cuda({7}, 11),
                          ref::random_cuda({3, 4, 5}, 12),
                          ref::random_cuda({6, 4}, 13).t().slice(0, 0, 4, 3)}) {
    const auto sum_ref = [](const std::vector<Values>& in) {
      double total = 0.0;
      for (const double v : in[0]) total += v;
      return Values{total};
    };
    const auto mean_ref = [&](const std::vector<Values>& in) {
      return Values{sum_ref(in)[0] / static_cast<double>(in[0].size())};
    };
    EXPECT_TRUE(gradcheck::check([](const auto& in) { return fg::sum(in[0]); }, sum_ref, {a}))
        << fg::describe(a);
    EXPECT_TRUE(gradcheck::check([](const auto& in) { return fg::mean(in[0]); }, mean_ref, {a}))
        << fg::describe(a);
  }
}

TEST(ReduceGrads, MeanAlongEachDimension) {
  SKIP_IF_NO_CUDA();
  for (const Tensor& a : inputs()) {
    for (std::int64_t d = 0; d < a.dim(); ++d) {
      const auto op = [&](const std::vector<Tensor>& in) { return fg::mean(in[0], d); };
      const auto reference = [&](const std::vector<Values>& in) {
        Values out = ref::reduce(in[0], a.shape(), d, add, 0.0);
        for (double& v : out) v /= static_cast<double>(a.size(d));
        return out;
      };
      EXPECT_TRUE(gradcheck::check(op, reference, {a})) << "dim " << d << " of " << fg::describe(a);
    }
  }
}

TEST(ReduceGrads, ScalarLossBackwardWithoutAGradient) {
  SKIP_IF_NO_CUDA();
  // The usual training pattern: a 0-dim loss, backward() without arguments.
  const Tensor x = ref::random_cuda({2, 5}, 20).requires_grad_();
  fg::mean(x * x).backward();
  const std::vector<float> xv = x.to_vector();
  std::vector<float> expected(xv.size());
  for (std::size_t i = 0; i < xv.size(); ++i) expected[i] = 2.0f * xv[i] / 10.0f;
  EXPECT_TRUE(ref::close(x.grad().to_vector(), expected, 1e-6, 1e-7));
}

TEST(ReduceGrads, SumToIsDifferentiable) {
  SKIP_IF_NO_CUDA();
  // Built from sum, so it needs no backward of its own; its gradient is an expand.
  const Tensor a = ref::random_cuda({4, 2, 3}, 30);
  const Shape to{2, 1};
  const auto op = [&](const std::vector<Tensor>& in) { return fg::sum_to(in[0], to); };
  const auto reference = [&](const std::vector<Values>& in) {
    Values out(2, 0.0);
    for (std::size_t i = 0; i < in[0].size(); ++i) out[i / 3 % 2] += in[0][i];
    return out;
  };
  EXPECT_TRUE(gradcheck::check(op, reference, {a}));
}

}  // namespace
