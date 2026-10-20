#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "flashgrad/core/ops.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "gradcheck.h"
#include "reference.h"

namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
using fg::Tensor;
using gradcheck::Values;

void check_matmul(const Tensor& a, const Tensor& b, const gradcheck::Options& options = {}) {
  const std::int64_t m = a.size(0), k = a.size(1), n = b.size(1);
  const auto op = [](const std::vector<Tensor>& in) { return fg::matmul(in[0], in[1]); };
  const auto reference = [&](const std::vector<Values>& in) {
    return ref::matmul(in[0], in[1], m, k, n);
  };
  EXPECT_TRUE(gradcheck::check(op, reference, {a, b}, options))
      << fg::describe(a) << " @ " << fg::describe(b);
}

TEST(MatmulGrads, SizesAroundTheTileSize) {
  SKIP_IF_NO_CUDA();
  std::uint64_t seed = 1;
  for (const auto& [m, k, n] : std::vector<std::array<std::int64_t, 3>>{
           {1, 1, 1}, {3, 4, 5}, {17, 3, 2}, {2, 33, 3}, {5, 2, 18}}) {
    check_matmul(ref::random_cuda({m, k}, seed), ref::random_cuda({k, n}, seed + 1));
    seed += 2;
  }
}

TEST(MatmulGrads, NonContiguousInputs) {
  SKIP_IF_NO_CUDA();
  check_matmul(ref::random_cuda({4, 3}, 10).t(), ref::random_cuda({4, 10}, 11).slice(1, 0, 10, 2));
}

TEST(MatmulGrads, OnlyOneInputRequiresGrad) {
  SKIP_IF_NO_CUDA();
  // The first layer of a network: the data doesn't require grad, the weights do.
  const Tensor x = ref::random_cuda({6, 4}, 20), w = ref::random_cuda({4, 3}, 21);
  check_matmul(x, w, {.requires_grad = {false, true}});
  check_matmul(x, w, {.requires_grad = {true, false}});
}

TEST(MatmulGrads, EmptyInnerDimension) {
  SKIP_IF_NO_CUDA();
  const fg::TensorOptions gpu{.device = fg::Device::cuda()};
  const Tensor a = fg::empty({3, 0}, gpu).requires_grad_();
  const Tensor b = fg::empty({0, 2}, gpu).requires_grad_();
  fg::matmul(a, b).backward(fg::ones({3, 2}, gpu));
  EXPECT_EQ(a.grad().shape(), a.shape());
  EXPECT_EQ(b.grad().shape(), b.shape());
}

TEST(MatmulGrads, LinearLayer) {
  SKIP_IF_NO_CUDA();
  // x @ w + bias: the bias gradient is the output gradient summed over the batch.
  const Tensor x = ref::random_cuda({5, 4}, 30);
  const Tensor w = ref::random_cuda({4, 3}, 31), bias = ref::random_cuda({3}, 32);
  const auto op = [](const std::vector<Tensor>& in) {
    return fg::matmul(in[0], in[1]) + in[2];
  };
  const auto reference = [](const std::vector<Values>& in) {
    Values out = ref::matmul(in[0], in[1], 5, 4, 3);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] += in[2][i % 3];
    return out;
  };
  EXPECT_TRUE(
      gradcheck::check(op, reference, {x, w, bias}, {.requires_grad = {false, true, true}}));
}

}  // namespace
