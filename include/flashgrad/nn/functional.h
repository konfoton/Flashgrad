#pragma once

#include <cstdint>

#include "flashgrad/core/tensor.h"

// Activation functions as free functions, like torch.nn.functional. Each one is composed of
// the ops in core/ops.h, so it runs on CUDA tensors and autograd differentiates it without a
// hand-written backward. The result is a new tensor with the shape of the input.
//
//   namespace F = flashgrad::nn::functional;
//   Tensor h = F::gelu(fg::matmul(x, w) + b);
namespace flashgrad::nn::functional {

// x * sigmoid(x), also called swish.
Tensor silu(const Tensor& x);

// x where x > 0, negative_slope * x elsewhere.
Tensor leaky_relu(const Tensor& x, float negative_slope = 0.01f);

// Hyperbolic tangent. Values and gradients stay finite for any finite x, e.g. tanh(±1000)
// is ±1.
Tensor tanh(const Tensor& x);

// exp(x) / sum(exp(x)) along `dim`; negative dims count from the end. Every slice along
// `dim` sums to 1, and large inputs don't overflow.
Tensor softmax(const Tensor& x, std::int64_t dim);

// GELU with the tanh approximation used by GPT-2:
//   0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
Tensor gelu(const Tensor& x);

}  // namespace flashgrad::nn::functional
