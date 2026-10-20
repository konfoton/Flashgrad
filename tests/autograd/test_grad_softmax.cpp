#include <gtest/gtest.h>

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

void check_log_softmax(const Tensor& a, std::int64_t dim) {
  const std::int64_t d = dim < 0 ? dim + a.dim() : dim;
  const auto op = [&](const std::vector<Tensor>& in) { return fg::log_softmax(in[0], dim); };
  const auto reference = [&](const std::vector<Values>& in) {
    return ref::log_softmax(in[0], a.shape(), d);
  };
  EXPECT_TRUE(gradcheck::check(op, reference, {a})) << "dim " << dim << " of " << fg::describe(a);
}

TEST(SoftmaxGrads, LogSoftmaxAlongEachDimension) {
  SKIP_IF_NO_CUDA();
  const Tensor a = ref::random_cuda({3, 4, 5}, 1);
  for (const std::int64_t dim : {0, 1, 2, -1}) check_log_softmax(a, dim);
  check_log_softmax(ref::random_cuda({7}, 2), 0);
  check_log_softmax(ref::random_cuda({1, 1}, 3), 1);  // a single class: the output is 0
}

TEST(SoftmaxGrads, NonContiguousInput) {
  SKIP_IF_NO_CUDA();
  const Tensor a = ref::random_cuda({6, 5}, 4).t().slice(1, 0, 6, 2);  // [5, 3]
  check_log_softmax(a, 0);
  check_log_softmax(a, 1);
}

TEST(SoftmaxGrads, LargeLogits) {
  SKIP_IF_NO_CUDA();
  // exp of these overflows in float32; the gradient must stay finite.
  check_log_softmax(ref::random_cuda({4, 6}, 5, [](float v) { return 100.0f * v; }), 1);
}

TEST(SoftmaxGrads, CrossEntropyGradientIsSoftmaxMinusOneHot) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  const Tensor logits = ref::random_cuda({4, 5}, 6).requires_grad_();
  const Tensor labels = fg::tensor({1, 0, 4, 2}, {4, 1}, gpu);
  const Tensor one_hot = fg::eq(labels, fg::arange(5.0).view({1, 5}).cuda());
  const Tensor loss = fg::mean(-fg::sum(fg::log_softmax(logits, 1) * one_hot, 1));
  loss.backward();
  const Tensor softmax = fg::exp(fg::log_softmax(logits.detach(), 1));
  const Tensor expected = (softmax - one_hot) / 4.0f;
  EXPECT_TRUE(ref::close(logits.grad().to_vector(), expected.to_vector(), 1e-5, 1e-7));
}

}  // namespace
