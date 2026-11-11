#include "flashgrad/nn/functional.h"

#include <cstdint>

#include "flashgrad/core/ops.h"
#include "flashgrad/core/tensor.h"

namespace flashgrad::nn::functional {

Tensor silu(const Tensor& x) { return x * sigmoid(x); }

Tensor leaky_relu(const Tensor& x, float negative_slope) {
  return relu(x) - negative_slope * relu(-x);
}

Tensor tanh(const Tensor& x) { return 2.0f * sigmoid(2.0f * x) - 1.0f; }

Tensor softmax(const Tensor& x, std::int64_t dim) { return exp(log_softmax(x, dim)); }

Tensor gelu(const Tensor& x) {
  constexpr float kSqrt2OverPi = 0.7978845608f;
  constexpr float kGeluCubic = 0.044715f;
  return 0.5f * x * (1.0f + tanh(kSqrt2OverPi * (x + kGeluCubic * x * x * x)));
}

}  // namespace flashgrad::nn::functional
