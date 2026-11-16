#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include "flashgrad/core/device.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/nn/activation.h"
#include "flashgrad/nn/module.h"

namespace flashgrad::nn {

// Layers applied one after another: forward(x) = layer_n(...layer_1(layer_0(x))). Each layer
// is a submodule named after its index, so parameters are named "0.weight", "2.bias", ...
//
//   Sequential model{std::make_shared<Linear>(784, 128), std::make_shared<ReLU>(),
//                    std::make_shared<Linear>(128, 10)};
//   model.push_back(std::make_shared<Softmax>(1));
//   Tensor probs = model(x);
class Sequential : public Module {
 public:
  Sequential() = default;
  // Appends the layers in order, as if by push_back.
  Sequential(std::initializer_list<std::shared_ptr<Module>> layers);

  // Appends `layer` (not null) as the submodule named std::to_string(size()).
  void push_back(const std::shared_ptr<Module>& layer);

  // Applies the layers in order; with no layers, returns x.
  Tensor forward(const Tensor& x) override;

  std::size_t size() const { return layers_.size(); }
  // The layer at `index`, which must be less than size().
  std::shared_ptr<Module> operator[](std::size_t index) const;

 private:
  std::vector<std::shared_ptr<Module>> layers_;
};

struct MLPOptions {
  Activation activation = Activation::kReLU;  // between consecutive Linear layers
  bool bias = true;                           // for every Linear layer
  Device device = Device::cpu();
};

// A multilayer perceptron: Linear layers with the given sizes and an activation between each
// two of them, but not after the last one. MLP({784, 128, 64, 10}) is
//   Linear(784, 128), ReLU, Linear(128, 64), ReLU, Linear(64, 10)
// as a Sequential, so the layers are named "0" to "4". `sizes` needs at least two entries.
class MLP : public Sequential {
 public:
  explicit MLP(const std::vector<std::int64_t>& sizes, MLPOptions options = {});
};

}  // namespace flashgrad::nn
