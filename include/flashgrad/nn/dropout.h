#pragma once

#include "flashgrad/core/tensor.h"
#include "flashgrad/nn/module.h"

namespace flashgrad::nn {

// Dropout: in training mode, zeroes each element independently with probability p and scales
// the kept ones by 1 / (1 - p), so the expected value of every element stays the same
// ("inverted dropout"). In evaluation mode, and when p == 0, forward returns its input
// unchanged (the same tensor). The random mask comes from the default generator.
class Dropout : public Module {
 public:
  // p must be in [0, 1).
  explicit Dropout(float p = 0.5f);

  Tensor forward(const Tensor& x) override;

  float p() const { return p_; }

 private:
  float p_;
};

}  // namespace flashgrad::nn
