#include "flashgrad/optim/sgd.h"

#include <cstddef>

#include "flashgrad/autograd/grad_mode.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"

namespace flashgrad::optim {

SGD::SGD(const std::vector<Tensor>& parameters, SGDOptions options)
    : Optimizer(parameters), options_(options), velocity_(parameters_.size()) {
  FG_CHECK(options.lr > 0.0f, "SGD: lr must be positive, got ", options.lr);
  FG_CHECK(options.momentum >= 0.0f && options.momentum < 1.0f,
           "SGD: momentum must be in [0, 1), got ", options.momentum);
  FG_CHECK(options.weight_decay >= 0.0f, "SGD: weight_decay must be non-negative, got ",
           options.weight_decay);
}

void SGD::step() {
  const NoGradGuard no_grad;
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    const Tensor& parameter = parameters_[i];
    Tensor grad = parameter.grad();
    if (!grad.defined()) continue;
    if (options_.weight_decay != 0.0f) grad = grad + options_.weight_decay * parameter;
    if (options_.momentum != 0.0f) {
      if (!velocity_[i].defined())
        velocity_[i] = grad;
      else
        velocity_[i] = velocity_[i] * options_.momentum + grad;
      grad = velocity_[i];
    }
    parameter.sub_(options_.lr * grad);
  }
}

}  // namespace flashgrad::optim
