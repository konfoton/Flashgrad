#include "flashgrad/optim/optimizer.h"

#include <cstddef>

#include "flashgrad/core/error.h"

namespace flashgrad::optim {

Optimizer::Optimizer(const std::vector<Tensor>& parameters) : parameters_(parameters) {
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    FG_CHECK(parameters[i].defined(), "Optimizer: parameter ", i, " is undefined");
    FG_CHECK(parameters[i].is_leaf() && parameters[i].requires_grad(), "Optimizer: parameter ",
             i, " ", describe(parameters[i]), " must be a leaf that requires grad");
  }
}

void Optimizer::zero_grad() const {
  for (const Tensor& parameter : parameters_) {
    parameter.set_grad(Tensor());
  }
}

}  // namespace flashgrad::optim
