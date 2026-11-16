#pragma once

#include <vector>

#include "flashgrad/core/tensor.h"

// Optimizers update parameters in place from their gradients, like torch.optim:
//
//   optim::SGD optimizer(model.parameters(), {.lr = 0.1f, .momentum = 0.9f});
//   for (...) {
//     Tensor loss = nn::cross_entropy(model(x), y);
//     optimizer.zero_grad();
//     loss.backward();
//     optimizer.step();
//   }
namespace flashgrad::optim {

// Base class of every optimizer. It keeps handles to the parameters (the same tensors as the
// model's, so updating them in place updates the model).
class Optimizer {
 public:
  // Every parameter must be a defined leaf that requires grad.
  explicit Optimizer(const std::vector<Tensor>& parameters);
  virtual ~Optimizer() = default;
  Optimizer(const Optimizer&) = delete;
  Optimizer& operator=(const Optimizer&) = delete;

  // Updates every parameter that has a gradient; parameters whose grad() is undefined are
  // skipped. The updates aren't recorded by autograd, and the gradients aren't changed.
  virtual void step() = 0;

  // Clears the gradient of every parameter (grad() becomes undefined).
  void zero_grad() const;

  const std::vector<Tensor>& parameters() const { return parameters_; }

 protected:
  std::vector<Tensor> parameters_;
};

}  // namespace flashgrad::optim
