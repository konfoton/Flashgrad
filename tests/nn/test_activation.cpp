#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "flashgrad/core/ops.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "flashgrad/nn/activation.h"
#include "flashgrad/nn/functional.h"
#include "reference.h"

// Activation modules: each must compute the same values as its function, and have no
// parameters.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
namespace nn = flashgrad::nn;
namespace F = flashgrad::nn::functional;
using fg::Tensor;

// `module` must give exactly the values of `expected` for a few inputs.
template <typename Expected>
void expect_same_as(nn::Module& module, Expected expected) {
  for (const Tensor& x : {ref::random_cuda({5}, 1), ref::random_cuda({3, 4}, 2),
                          ref::random_cuda({2, 3, 4}, 3)}) {
    EXPECT_EQ(module(x).to_vector(), expected(x).to_vector()) << fg::describe(x);
  }
  EXPECT_TRUE(module.parameters().empty());
}

TEST(Activation, ReLU) {
  SKIP_IF_NO_CUDA();
  nn::ReLU m;
  expect_same_as(m, [](const Tensor& x) { return fg::relu(x); });
}

TEST(Activation, Sigmoid) {
  SKIP_IF_NO_CUDA();
  nn::Sigmoid m;
  expect_same_as(m, [](const Tensor& x) { return fg::sigmoid(x); });
}

TEST(Activation, Tanh) {
  SKIP_IF_NO_CUDA();
  nn::Tanh m;
  expect_same_as(m, [](const Tensor& x) { return F::tanh(x); });
}

TEST(Activation, GELU) {
  SKIP_IF_NO_CUDA();
  nn::GELU m;
  expect_same_as(m, [](const Tensor& x) { return F::gelu(x); });
}

TEST(Activation, SiLU) {
  SKIP_IF_NO_CUDA();
  nn::SiLU m;
  expect_same_as(m, [](const Tensor& x) { return F::silu(x); });
}

TEST(Activation, LeakyReLU) {
  SKIP_IF_NO_CUDA();
  nn::LeakyReLU default_slope;
  EXPECT_EQ(default_slope.negative_slope(), 0.01f);
  expect_same_as(default_slope, [](const Tensor& x) { return F::leaky_relu(x, 0.01f); });
  nn::LeakyReLU steep(0.3f);
  expect_same_as(steep, [](const Tensor& x) { return F::leaky_relu(x, 0.3f); });
}

TEST(Activation, Softmax) {
  SKIP_IF_NO_CUDA();
  nn::Softmax last(-1);
  EXPECT_EQ(last.dim(), -1);
  expect_same_as(last, [](const Tensor& x) { return F::softmax(x, -1); });
  nn::Softmax first(0);
  expect_same_as(first, [](const Tensor& x) { return F::softmax(x, 0); });
}

TEST(Activation, MakeActivationBuildsTheRequestedModule) {
  SKIP_IF_NO_CUDA();
  const Tensor x = ref::random_cuda({4, 6}, 4);
  const auto check = [&](nn::Activation a, const Tensor& expected) {
    const std::shared_ptr<nn::Module> m = nn::make_activation(a);
    ASSERT_NE(m, nullptr);
    EXPECT_EQ((*m)(x).to_vector(), expected.to_vector());
  };
  check(nn::Activation::kReLU, fg::relu(x));
  check(nn::Activation::kSigmoid, fg::sigmoid(x));
  check(nn::Activation::kTanh, F::tanh(x));
  check(nn::Activation::kGELU, F::gelu(x));
  check(nn::Activation::kSiLU, F::silu(x));
}

TEST(Activation, MakeActivationReturnsANewModuleEachTime) {
  const auto a = nn::make_activation(nn::Activation::kReLU);
  const auto b = nn::make_activation(nn::Activation::kReLU);
  EXPECT_NE(a.get(), b.get());
}

}  // namespace
