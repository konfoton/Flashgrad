#include "flashgrad/nn/loss.h"

#include <cstdint>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"

namespace flashgrad::nn {

Tensor mse_loss(const Tensor& input, const Tensor& target) {
  FG_CHECK(input.shape() == target.shape(), "mse_loss: shapes differ, ", describe(input), " and ",
           describe(target));
  const Tensor diff = input - target;
  return mean(diff * diff);
}

Tensor cross_entropy(const Tensor& logits, const Tensor& target) {
  FG_CHECK(logits.dim() == 2, "cross_entropy: logits must be [batch, classes], got ",
           describe(logits));
  FG_CHECK(target.dim() == 1 && target.size(0) == logits.size(0), "cross_entropy: target must be [",
           logits.size(0), "], got ", describe(target));
  const std::int64_t batch = logits.size(0);
  const std::int64_t classes = logits.size(1);
  const Tensor indexes = arange(static_cast<double>(classes), {.device = logits.device()});
  const Tensor one_hot = eq(target.view({batch, 1}), indexes);
  return mean(-sum(log_softmax(logits, 1) * one_hot, 1));
}

}  // namespace flashgrad::nn
