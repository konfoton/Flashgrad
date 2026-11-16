#pragma once

#include <cstdint>
#include <vector>

#include "flashgrad/core/tensor.h"
#include "flashgrad/optim/optimizer.h"

namespace flashgrad::optim {

struct AdamOptions {
  float lr = 1e-3f;                     // > 0
  float beta1 = 0.9f;                   // in [0, 1)
  float beta2 = 0.999f;                 // in [0, 1)
  float eps = 1e-8f;                    // > 0
  float weight_decay = 0.0f;            // >= 0
  bool decoupled_weight_decay = false;  // true gives AdamW
};

// Adam (Kingma & Ba), as in PyTorch. On step t = 1, 2, ..., for each parameter p with
// gradient grad:
//
//   g = grad + weight_decay * p              (Adam: L2 penalty in the gradient)
//   p = p * (1 - lr * weight_decay)          (AdamW instead: decoupled decay, g = grad)
//   m = beta1 * m + (1 - beta1) * g          (m and v start at zero)
//   v = beta2 * v + (1 - beta2) * g^2
//   m_hat = m / (1 - beta1^t),  v_hat = v / (1 - beta2^t)
//   p = p - lr * m_hat / (sqrt(v_hat) + eps)
class Adam : public Optimizer {
 public:
  Adam(const std::vector<Tensor>& parameters, AdamOptions options);

  void step() override;

  const AdamOptions& options() const { return options_; }
  std::int64_t steps() const { return steps_; }  // how many times step() was called

 private:
  AdamOptions options_;
  std::int64_t steps_ = 0;
  std::vector<Tensor> m_;  // first moments, one per parameter; undefined until used
  std::vector<Tensor> v_;  // second moments
};

}  // namespace flashgrad::optim
