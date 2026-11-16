#include "flashgrad/nn/sequential.h"

#include <string>

#include "flashgrad/core/error.h"
#include "flashgrad/nn/linear.h"

namespace flashgrad::nn {

Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> layers) {
  for (const auto& layer : layers) {
    push_back(layer);
  }
}

void Sequential::push_back(const std::shared_ptr<Module>& layer) {
  register_module(std::to_string(layers_.size()), layer);
  layers_.push_back(layer);
}

Tensor Sequential::forward(const Tensor& x) {
  Tensor out = x;
  for (const auto& layer : layers_) {
    out = layer->forward(out);
  }
  return out;
}

std::shared_ptr<Module> Sequential::operator[](std::size_t index) const {
  FG_CHECK(index < layers_.size(), "Sequential: index ", index, " out of range for ",
           layers_.size(), " layers");
  return layers_[index];
}

MLP::MLP(const std::vector<std::int64_t>& sizes, MLPOptions options) {
  FG_CHECK(sizes.size() >= 2, "MLP: needs at least two sizes, got ", sizes.size());
  const LinearOptions linear_options{.bias = options.bias, .device = options.device};
  for (std::size_t i = 0; i + 1 < sizes.size(); ++i) {
    push_back(std::make_shared<Linear>(sizes[i], sizes[i + 1], linear_options));
    if (i + 2 != sizes.size()) { push_back(make_activation(options.activation)); }
  }
}

}  // namespace flashgrad::nn
