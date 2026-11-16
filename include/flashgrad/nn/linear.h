#pragma once

#include <cstdint>

#include "flashgrad/core/device.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/nn/module.h"

namespace flashgrad::nn {

struct LinearOptions {
  bool bias = true;                // whether the layer has a bias
  Device device = Device::cpu();   // where the parameters are created
};

// Fully connected layer: y = x @ weight + bias, for x of shape [batch, in_features].
//
// weight has shape [in_features, out_features] (the transpose of PyTorch's layout, so forward
// needs no transpose) and bias has shape [out_features]. Both start uniform on [-k, k) with
// k = 1 / sqrt(in_features), like PyTorch's default; the values come from the default
// generator (flashgrad::manual_seed makes them reproducible).
//
//   auto fc = std::make_shared<Linear>(784, 128, LinearOptions{.device = Device::cuda()});
//   Tensor h = (*fc)(x);   // x [64, 784] -> h [64, 128]
class Linear : public Module {
 public:
  // in_features and out_features must be positive.
  Linear(std::int64_t in_features, std::int64_t out_features, LinearOptions options = {});

  // x must be 2-D with x.size(1) == in_features; the result is [x.size(0), out_features].
  Tensor forward(const Tensor& x) override;

  std::int64_t in_features() const { return in_features_; }
  std::int64_t out_features() const { return out_features_; }
  const Tensor& weight() const { return weight_; }
  const Tensor& bias() const { return bias_; }  // undefined without a bias

 private:
  static Tensor uniform(Shape shape, float k, Device device);

  std::int64_t in_features_;
  std::int64_t out_features_;
  Tensor weight_;
  Tensor bias_;
};

}  // namespace flashgrad::nn
