#pragma once

#include "flashgrad/core/tensor.h"

// Loss functions. Each returns a 0-dim tensor (a single number) averaged over the batch, so
// loss.backward() works without passing a gradient.
namespace flashgrad::nn {

// Mean squared error: mean((input - target)^2) over all elements. `input` and `target` must
// have exactly the same shape (no broadcasting, which would silently compare the wrong
// elements).
Tensor mse_loss(const Tensor& input, const Tensor& target);

// Cross-entropy of raw scores against class indices, averaged over the batch:
//   mean over n of  -log(softmax(logits[n])[target[n]])
// `logits` is [batch, classes] (no softmax applied); `target` is [batch] and holds class
// indices 0 .. classes - 1 stored as floats, since every tensor is float32. Indices outside
// that range aren't checked (that would need a copy to the host) and contribute 0.
Tensor cross_entropy(const Tensor& logits, const Tensor& target);

}  // namespace flashgrad::nn
