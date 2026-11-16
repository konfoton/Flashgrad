#include "flashgrad/nn/dropout.h"

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"

namespace flashgrad::nn {

Dropout::Dropout(float p) : p_(p) {
  FG_CHECK(p >= 0.0f && p < 1.0f, "Dropout: p must be in [0, 1), got ", p);
}

Tensor Dropout::forward(const Tensor& x) {
  if (!is_training() || p_ == 0.0f) return x;
  const float keep = 1.0f - p_;
  const Tensor mask =
      lt(rand(x.shape(), {.device = x.device()}), full({}, keep, {.device = x.device()}));
  return x * mask / keep;
}

}  // namespace flashgrad::nn
