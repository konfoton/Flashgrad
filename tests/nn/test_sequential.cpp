#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "flashgrad/autograd/grad_mode.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "flashgrad/nn/activation.h"
#include "flashgrad/nn/functional.h"
#include "flashgrad/nn/linear.h"
#include "flashgrad/nn/sequential.h"
#include "reference.h"

// nn::Sequential and nn::MLP. Structure tests run on the CPU; forward and training need a GPU.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
namespace nn = flashgrad::nn;
using fg::Tensor;

std::vector<std::string> names(const nn::Module& module) {
  std::vector<std::string> out;
  for (const auto& [name, tensor] : module.named_parameters()) out.push_back(name);
  return out;
}

// ---- Sequential: structure ------------------------------------------------------------

TEST(Sequential, EmptySequentialReturnsItsInput) {
  nn::Sequential seq;
  EXPECT_EQ(seq.size(), 0u);
  const Tensor x = fg::ones({2, 3});
  EXPECT_TRUE(seq(x).is_same(x));
  EXPECT_TRUE(seq.parameters().empty());
}

TEST(Sequential, PushBackNamesLayersByIndex) {
  nn::Sequential seq;
  const auto fc = std::make_shared<nn::Linear>(2, 3);
  seq.push_back(fc);
  seq.push_back(std::make_shared<nn::ReLU>());
  seq.push_back(std::make_shared<nn::Linear>(3, 1));
  EXPECT_EQ(seq.size(), 3u);
  EXPECT_EQ(names(seq), (std::vector<std::string>{"0.weight", "0.bias", "2.weight", "2.bias"}));
  EXPECT_EQ(seq[0].get(), fc.get());  // the same object, not a copy
}

TEST(Sequential, InitializerListMatchesPushBack) {
  const nn::Sequential seq{std::make_shared<nn::Linear>(2, 3), std::make_shared<nn::Tanh>(),
                           std::make_shared<nn::Linear>(3, 1, nn::LinearOptions{.bias = false})};
  EXPECT_EQ(seq.size(), 3u);
  EXPECT_EQ(names(seq), (std::vector<std::string>{"0.weight", "0.bias", "2.weight"}));
}

TEST(Sequential, NestedSequentialsPrefixNames) {
  const auto inner = std::make_shared<nn::Sequential>();
  inner->push_back(std::make_shared<nn::Linear>(2, 2));
  nn::Sequential outer;
  outer.push_back(std::make_shared<nn::ReLU>());
  outer.push_back(inner);
  EXPECT_EQ(names(outer), (std::vector<std::string>{"1.0.weight", "1.0.bias"}));
}

TEST(Sequential, RejectsNullLayersAndOutOfRangeIndices) {
  nn::Sequential seq{std::make_shared<nn::ReLU>()};
  EXPECT_THROW(seq.push_back(nullptr), fg::Error);
  EXPECT_EQ(seq.size(), 1u);
  EXPECT_THROW(seq[1], fg::Error);
}

TEST(Sequential, TrainAndEvalReachTheLayers) {
  nn::Sequential seq{std::make_shared<nn::Linear>(2, 2), std::make_shared<nn::ReLU>()};
  seq.eval();
  EXPECT_FALSE(seq[0]->is_training());
  EXPECT_FALSE(seq[1]->is_training());
  seq.train();
  EXPECT_TRUE(seq[0]->is_training());
}

// ---- Sequential: forward --------------------------------------------------------------

TEST(Sequential, ForwardAppliesTheLayersInOrder) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  const auto fc1 = std::make_shared<nn::Linear>(4, 6, nn::LinearOptions{.device = gpu});
  const auto fc2 = std::make_shared<nn::Linear>(6, 3, nn::LinearOptions{.device = gpu});
  nn::Sequential seq{fc1, std::make_shared<nn::ReLU>(), fc2};
  const Tensor x = ref::random_cuda({5, 4}, 1);
  const Tensor expected = (*fc2)(fg::relu((*fc1)(x)));
  EXPECT_EQ(seq(x).to_vector(), expected.to_vector());
}

// ---- MLP ------------------------------------------------------------------------------

TEST(MLP, LayersAndNames) {
  const nn::MLP mlp({4, 8, 6, 2});
  ASSERT_EQ(mlp.size(), 5u);  // Linear, act, Linear, act, Linear
  for (const std::size_t i : {0u, 2u, 4u}) {
    EXPECT_NE(std::dynamic_pointer_cast<nn::Linear>(mlp[i]), nullptr) << "layer " << i;
  }
  for (const std::size_t i : {1u, 3u}) {
    EXPECT_NE(std::dynamic_pointer_cast<nn::ReLU>(mlp[i]), nullptr) << "layer " << i;
  }
  const auto last = std::dynamic_pointer_cast<nn::Linear>(mlp[4]);
  EXPECT_EQ(last->in_features(), 6);
  EXPECT_EQ(last->out_features(), 2);
  EXPECT_EQ(names(mlp), (std::vector<std::string>{"0.weight", "0.bias", "2.weight", "2.bias",
                                                  "4.weight", "4.bias"}));
}

TEST(MLP, SingleLayerHasNoActivation) {
  const nn::MLP mlp({3, 2});
  ASSERT_EQ(mlp.size(), 1u);
  EXPECT_NE(std::dynamic_pointer_cast<nn::Linear>(mlp[0]), nullptr);
}

TEST(MLP, Options) {
  const nn::MLP mlp({3, 5, 2}, {.activation = nn::Activation::kGELU, .bias = false});
  ASSERT_EQ(mlp.size(), 3u);
  EXPECT_NE(std::dynamic_pointer_cast<nn::GELU>(mlp[1]), nullptr);
  EXPECT_EQ(names(mlp), (std::vector<std::string>{"0.weight", "2.weight"}));
}

TEST(MLP, RejectsFewerThanTwoSizes) {
  EXPECT_THROW(nn::MLP({}), fg::Error);
  EXPECT_THROW(nn::MLP({4}), fg::Error);
  EXPECT_THROW(nn::MLP({4, 0}), fg::Error);  // from Linear
}

TEST(MLP, ParametersOnTheRequestedDevice) {
  SKIP_IF_NO_CUDA();
  const nn::MLP mlp({3, 4, 2}, {.device = fg::Device::cuda()});
  for (const Tensor& p : mlp.parameters()) EXPECT_TRUE(p.device().is_cuda());
}

TEST(MLP, LearnsTheQuadrantXor) {
  SKIP_IF_NO_CUDA();
  // The task of examples/autograd_basics.cpp: the label is the sign of x0 * x1.
  const fg::TensorOptions gpu{.device = fg::Device::cuda()};
  fg::manual_seed(0);
  constexpr std::int64_t n = 512;
  const Tensor x = fg::randn({n, 2}, gpu);
  const Tensor label = fg::lt(fg::zeros({}, gpu), x.select(1, 0) * x.select(1, 1));
  const Tensor one_hot = fg::eq(label.view({n, 1}), fg::arange(2.0, gpu).view({1, 2}));

  nn::MLP model({2, 32, 2}, {.device = gpu.device});
  constexpr float kLearningRate = 0.5f;
  // Stops as soon as the accuracy is good: kernels are slow in Debug builds.
  float accuracy = 0.0f;
  for (int step = 0; step < 1500 && accuracy <= 0.95f; ++step) {
    const Tensor logits = model(x);
    if (step % 50 == 0) {
      const Tensor out = logits.detach();
      const Tensor is_max = fg::eq(out, fg::max(out, 1, /*keepdim=*/true));
      accuracy = fg::sum(is_max * one_hot).item() / static_cast<float>(n);
    }
    const Tensor loss = fg::mean(-fg::sum(fg::log_softmax(logits, 1) * one_hot, 1));
    model.zero_grad();
    loss.backward();
    const fg::NoGradGuard no_grad;
    for (const Tensor& p : model.parameters()) p.sub_(p.grad() * kLearningRate);
  }
  EXPECT_GT(accuracy, 0.9f);
}

}  // namespace
