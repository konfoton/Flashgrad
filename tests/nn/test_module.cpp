#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "flashgrad/autograd/grad_mode.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "flashgrad/nn/module.h"
#include "reference.h"

// The nn::Module base class: parameter and submodule registration, naming, train/eval mode
// and gradients. The registration tests use CPU tensors and don't need a GPU.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
namespace nn = flashgrad::nn;
using fg::Tensor;

// ---- Modules used by the tests -------------------------------------------------------

// y = x * weight + bias, elementwise; weight starts at 2 and bias at 0.5.
class Affine : public nn::Module {
 public:
  explicit Affine(std::int64_t n, fg::TensorOptions options = {})
      : weight_(register_parameter("weight", fg::full({n}, 2.0, options))),
        bias_(register_parameter("bias", fg::full({n}, 0.5, options))) {}

  Tensor forward(const Tensor& x) override { return x * weight_ + bias_; }

  const Tensor& weight() const { return weight_; }
  const Tensor& bias() const { return bias_; }

 private:
  Tensor weight_;
  Tensor bias_;
};

// scale * second(first(x)): an own parameter and two submodules.
class Block : public nn::Module {
 public:
  explicit Block(std::int64_t n, fg::TensorOptions options = {})
      : scale_(register_parameter("scale", fg::ones({}, options))),
        first_(std::make_shared<Affine>(n, options)),
        second_(std::make_shared<Affine>(n, options)) {
    register_module("first", first_);
    register_module("second", second_);
  }

  Tensor forward(const Tensor& x) override { return (*second_)((*first_)(x)) * scale_; }

  const Affine& first() const { return *first_; }
  const Affine& second() const { return *second_; }

 private:
  Tensor scale_;
  std::shared_ptr<Affine> first_;
  std::shared_ptr<Affine> second_;
};

// A submodule registered before an own parameter: own parameters still come first.
class Model : public nn::Module {
 public:
  explicit Model(std::int64_t n, fg::TensorOptions options = {})
      : block_(std::make_shared<Block>(n, options)) {
    register_module("block", block_);
    gain_ = register_parameter("gain", fg::ones({n}, options));
  }

  Tensor forward(const Tensor& x) override { return (*block_)(x) * gain_; }

  const Block& block() const { return *block_; }

 private:
  std::shared_ptr<Block> block_;
  Tensor gain_;
};

// Exposes the protected registration methods so tests can call them directly.
class Registrar : public nn::Module {
 public:
  using nn::Module::register_module;
  using nn::Module::register_parameter;

  Tensor forward(const Tensor& x) override { return x; }
};

std::vector<std::string> names(const nn::Module& module) {
  std::vector<std::string> out;
  for (const auto& [name, tensor] : module.named_parameters()) out.push_back(name);
  return out;
}

// ---- Registration ---------------------------------------------------------------------

TEST(Module, RegisteredParametersAreLeavesThatRequireGrad) {
  const Affine affine(3);
  const std::vector<Tensor> params = affine.parameters();
  ASSERT_EQ(params.size(), 2u);
  for (const Tensor& p : params) {
    EXPECT_TRUE(p.requires_grad());
    EXPECT_TRUE(p.is_leaf());
  }
  EXPECT_EQ(affine.weight().to_vector(), std::vector<float>({2.0f, 2.0f, 2.0f}));
}

TEST(Module, ParametersAreHandlesToTheModulesTensors) {
  const Affine affine(3);
  const std::vector<Tensor> params = affine.parameters();
  ASSERT_EQ(params.size(), 2u);
  EXPECT_TRUE(params[0].is_same(affine.weight()));
  EXPECT_TRUE(params[1].is_same(affine.bias()));
  {
    const fg::NoGradGuard no_grad;
    params[0].fill_(7.0);
  }
  EXPECT_EQ(affine.weight().to_vector(), std::vector<float>({7.0f, 7.0f, 7.0f}));
}

TEST(Module, NamedParametersOfSubmodulesArePrefixed) {
  const Block block(2);
  EXPECT_EQ(names(block), (std::vector<std::string>{"scale", "first.weight", "first.bias",
                                                     "second.weight", "second.bias"}));
}

TEST(Module, OwnParametersComeBeforeSubmodules) {
  const Model model(2);
  EXPECT_EQ(names(model),
            (std::vector<std::string>{"gain", "block.scale", "block.first.weight",
                                      "block.first.bias", "block.second.weight",
                                      "block.second.bias"}));
}

TEST(Module, ParametersMatchNamedParameters) {
  const Model model(2);
  const std::vector<Tensor> params = model.parameters();
  const auto named = model.named_parameters();
  ASSERT_EQ(params.size(), named.size());
  for (std::size_t i = 0; i < params.size(); ++i) {
    EXPECT_TRUE(params[i].is_same(named[i].second)) << named[i].first;
  }
  EXPECT_TRUE(named[2].second.is_same(model.block().first().weight()));
}

TEST(Module, ModuleWithoutParameters) {
  const Registrar empty;
  EXPECT_TRUE(empty.parameters().empty());
  EXPECT_TRUE(empty.named_parameters().empty());
}

TEST(Module, RejectsInvalidRegistrations) {
  Registrar r;
  r.register_parameter("w", fg::ones({2}));
  r.register_module("child", std::make_shared<Registrar>());

  EXPECT_THROW(r.register_parameter("", fg::ones({2})), fg::Error);
  EXPECT_THROW(r.register_parameter("a.b", fg::ones({2})), fg::Error);
  EXPECT_THROW(r.register_parameter("w", fg::ones({2})), fg::Error);      // taken
  EXPECT_THROW(r.register_parameter("child", fg::ones({2})), fg::Error);  // taken by a module
  EXPECT_THROW(r.register_parameter("u", Tensor()), fg::Error);           // undefined

  EXPECT_THROW(r.register_module("", std::make_shared<Registrar>()), fg::Error);
  EXPECT_THROW(r.register_module("x.y", std::make_shared<Registrar>()), fg::Error);
  EXPECT_THROW(r.register_module("child", std::make_shared<Registrar>()), fg::Error);
  EXPECT_THROW(r.register_module("w", std::make_shared<Registrar>()), fg::Error);
  EXPECT_THROW(r.register_module("null", nullptr), fg::Error);

  EXPECT_EQ(names(r), (std::vector<std::string>{"w"}));  // failed calls registered nothing
}

TEST(Module, RejectsParametersThatAreNotLeaves) {
  SKIP_IF_NO_CUDA();
  Registrar r;
  const Tensor leaf = fg::ones({2}, {.device = fg::Device::cuda()}).requires_grad_();
  EXPECT_THROW(r.register_parameter("computed", leaf * 2.0f), fg::Error);
}

// ---- Training mode --------------------------------------------------------------------

TEST(Module, TrainAndEvalReachEverySubmodule) {
  Model model(2);
  const auto all_in_mode = [&](bool training) {
    return model.is_training() == training && model.block().is_training() == training &&
           model.block().first().is_training() == training &&
           model.block().second().is_training() == training;
  };
  EXPECT_TRUE(all_in_mode(true));
  model.eval();
  EXPECT_TRUE(all_in_mode(false));
  model.train();
  EXPECT_TRUE(all_in_mode(true));
  model.train(false);
  EXPECT_TRUE(all_in_mode(false));
}

// ---- Forward and gradients ------------------------------------------------------------

TEST(Module, ForwardThroughABasePointer) {
  SKIP_IF_NO_CUDA();
  const fg::Device gpu = fg::Device::cuda();
  const std::shared_ptr<nn::Module> module =
      std::make_shared<Affine>(3, fg::TensorOptions{.device = gpu});
  const Tensor x = fg::tensor({1.0f, -2.0f, 0.0f}, gpu);
  EXPECT_EQ((*module)(x).to_vector(), std::vector<float>({2.5f, -3.5f, 0.5f}));
  EXPECT_EQ(module->forward(x).to_vector(), std::vector<float>({2.5f, -3.5f, 0.5f}));
}

TEST(Module, ZeroGradClearsEveryGradient) {
  SKIP_IF_NO_CUDA();
  Model model(3, {.device = fg::Device::cuda()});
  fg::sum(model(ref::random_cuda({4, 3}, 1))).backward();
  for (const auto& [name, p] : model.named_parameters()) {
    EXPECT_TRUE(p.grad().defined()) << name;
  }
  model.zero_grad();
  for (const auto& [name, p] : model.named_parameters()) {
    EXPECT_FALSE(p.grad().defined()) << name;
  }
}

TEST(Module, GradientDescentThroughParametersFitsTheTarget) {
  SKIP_IF_NO_CUDA();
  // Learns y = 3x - 1 starting from weight 2 and bias 0.5.
  Affine model(4, {.device = fg::Device::cuda()});
  const Tensor x = ref::random_cuda({64, 4}, 2);
  const Tensor target = x * 3.0f - 1.0f;
  constexpr float kLearningRate = 0.5f;
  float loss_value = 0.0f;
  for (int step = 0; step < 200; ++step) {
    const Tensor diff = model(x) - target;
    const Tensor loss = fg::mean(diff * diff);
    loss_value = loss.item();
    model.zero_grad();
    loss.backward();
    const fg::NoGradGuard no_grad;
    for (const Tensor& p : model.parameters()) p.sub_(p.grad() * kLearningRate);
  }
  EXPECT_LT(loss_value, 1e-4f);
  EXPECT_TRUE(ref::close(model.weight().to_vector(), std::vector<float>(4, 3.0f), 1e-3, 1e-3));
  EXPECT_TRUE(ref::close(model.bias().to_vector(), std::vector<float>(4, -1.0f), 1e-3, 1e-3));
}

}  // namespace
