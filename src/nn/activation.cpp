#include "flashgrad/nn/activation.h"

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/nn/functional.h"

namespace flashgrad::nn {

Tensor ReLU::forward(const Tensor& x) { return relu(x); }

Tensor Sigmoid::forward(const Tensor& x) { return sigmoid(x); }

Tensor Tanh::forward(const Tensor& x) { return functional::tanh(x); }

Tensor GELU::forward(const Tensor& x) { return functional::gelu(x); }

Tensor SiLU::forward(const Tensor& x) { return functional::silu(x); }

Tensor LeakyReLU::forward(const Tensor& x) { return functional::leaky_relu(x, negative_slope_); }

Tensor Softmax::forward(const Tensor& x) { return functional::softmax(x, dim_); }

std::shared_ptr<Module> make_activation(Activation activation) {
  switch (activation) {
    case Activation::kReLU:
      return std::make_shared<ReLU>();
    case Activation::kSigmoid:
      return std::make_shared<Sigmoid>();
    case Activation::kTanh:
      return std::make_shared<Tanh>();
    case Activation::kGELU:
      return std::make_shared<GELU>();
    case Activation::kSiLU:
      return std::make_shared<SiLU>();
  }
  FG_THROW("make_activation: unknown activation");
}

}  // namespace flashgrad::nn
