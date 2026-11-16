#include "flashgrad/nn/linear.h"

#include <cmath>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"

namespace flashgrad::nn {

Linear::Linear(std::int64_t in_features, std::int64_t out_features, LinearOptions options)
    : in_features_(in_features), out_features_(out_features) {
  FG_CHECK(in_features > 0 && out_features > 0, "Linear: sizes must be positive, got ", in_features,
           " and ", out_features);
  const float k = 1.0f / std::sqrt(static_cast<float>(in_features));
  Shape weight_shape = std::vector<std::int64_t>{in_features, out_features};
  weight_ = register_parameter("weight", uniform(weight_shape, k, options.device));
  if (options.bias) {
    Shape bias_shape = std::vector<std::int64_t>{out_features};
    bias_ = register_parameter("bias", uniform(bias_shape, k, options.device));
  }
}

Tensor Linear::forward(const Tensor& x) {
  FG_CHECK(x.dim() == 2 && x.size(1) == in_features_, "Linear: expected input [batch, ",
           in_features_, "], got ", describe(x));
  if (bias_.defined()) {
    return matmul(x, weight_) + bias_;
  }
  return matmul(x, weight_);
}

Tensor Linear::uniform(Shape shape, float k, Device device) {
  std::vector<float> values = rand(shape).to_vector();
  for (float& v : values) {
    v = 2.0f * k * v - k;
  }
  return tensor(values, shape, device);
}

}  // namespace flashgrad::nn
