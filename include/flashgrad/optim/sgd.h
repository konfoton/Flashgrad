#pragma once

#include <vector>

#include "flashgrad/core/tensor.h"
#include "flashgrad/optim/optimizer.h"

namespace flashgrad::optim {

struct SGDOptions {
  float lr = 0.01f;            // learning rate, > 0
  float momentum = 0.0f;       // in [0, 1); 0 turns momentum off
  float weight_decay = 0.0f;   // L2 penalty, >= 0
};

// Stochastic gradient descent with optional momentum and weight decay, as in PyTorch. For each
// parameter p with gradient grad:
//
//   g = grad + weight_decay * p
//   v = g                    on the first step (when momentum > 0)
//   v = momentum * v + g     on later steps
//   p = p - lr * v           (p - lr * g without momentum)
class SGD : public Optimizer {
 public:
  SGD(const std::vector<Tensor>& parameters, SGDOptions options);

  void step() override;

  const SGDOptions& options() const { return options_; }

 private:
  SGDOptions options_;
  std::vector<Tensor> velocity_;  // one per parameter; undefined until its first step
};

}  // namespace flashgrad::optim
