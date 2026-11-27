#include "flashgrad/optim/adam.h"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "cuda/ops.h"
#include "flashgrad/autograd/grad_mode.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"

namespace flashgrad::optim {

namespace {

// Elementwise square root of a tensor with non-negative values. ops.h has no sqrt kernel, so
// it is computed as exp(0.5 * log(x)); for x == 0, log gives -inf and exp(-inf) gives 0.
Tensor sqrt_nonnegative(const Tensor& x) { return exp(log(x) * 0.5f); }

}  // namespace

Adam::Adam(const std::vector<Tensor>& parameters, AdamOptions options)
    : Optimizer(parameters), options_(options), m_(parameters_.size()), v_(parameters_.size()) {
  FG_CHECK(options.lr > 0.0f, "Adam: lr must be positive, got ", options.lr);
  FG_CHECK(options.beta1 >= 0.0f && options.beta1 < 1.0f, "Adam: beta1 must be in [0, 1), got ",
           options.beta1);
  FG_CHECK(options.beta2 >= 0.0f && options.beta2 < 1.0f, "Adam: beta2 must be in [0, 1), got ",
           options.beta2);
  FG_CHECK(options.eps > 0.0f, "Adam: eps must be positive, got ", options.eps);
  FG_CHECK(options.weight_decay >= 0.0f, "Adam: weight_decay must be non-negative, got ",
           options.weight_decay);
}

void Adam::step() {
  const NoGradGuard no_grad;
  ++steps_;
  const auto correction1 =
      static_cast<float>(1.0f - std::pow(options_.beta1, static_cast<double>(steps_)));
  const auto correction2 =
      static_cast<float>(1.0f - std::pow(options_.beta2, static_cast<double>(steps_)));

  std::vector<cuda::detail::AdamTensors> fused;
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    const Tensor& parameter = parameters_[i];
    Tensor grad = parameter.grad();
    if (!grad.defined()) continue;
    if (!options_.fused || !parameter.is_contiguous()) {
      step_unfused(i, std::move(grad), correction1, correction2);
      continue;
    }
    FG_CHECK(parameter.device().is_cuda(), "Adam: fused step needs CUDA parameters, got ",
             describe(parameter), " (parameter ", i, ")");
    FG_CHECK(fused.empty() || parameter.device() == fused.front().param.device(),
             "Adam: fused step needs all parameters on one device, got ", describe(parameter),
             " and ", describe(fused.front().param));
    if (!m_[i].defined()) m_[i] = zeros_like(parameter);
    if (!v_[i].defined()) v_[i] = zeros_like(parameter);
    fused.push_back({parameter, grad.contiguous(), m_[i], v_[i]});
  }
  cuda::detail::adam_step(fused, {.lr = options_.lr,
                                  .beta1 = options_.beta1,
                                  .beta2 = options_.beta2,
                                  .eps = options_.eps,
                                  .weight_decay = options_.weight_decay,
                                  .decoupled_weight_decay = options_.decoupled_weight_decay,
                                  .step = steps_});
}

// The same update composed from elementwise ops: about a dozen kernels and temporaries.
void Adam::step_unfused(std::size_t i, Tensor grad, float correction1, float correction2) {
  const Tensor& parameter = parameters_[i];
  if (options_.weight_decay != 0.0f) {
    if (options_.decoupled_weight_decay)
      parameter.mul_(1.0f - static_cast<double>(options_.lr) * options_.weight_decay);
    else
      grad = grad + options_.weight_decay * parameter;
  }
  if (!m_[i].defined()) m_[i] = zeros_like(parameter);
  if (!v_[i].defined()) v_[i] = zeros_like(parameter);
  m_[i] = m_[i] * options_.beta1 + grad * (1.0f - options_.beta1);
  v_[i] = v_[i] * options_.beta2 + grad * grad * (1.0f - options_.beta2);
  const Tensor denominator = sqrt_nonnegative(v_[i] / correction2) + options_.eps;
  parameter.sub_(m_[i] / correction1 / denominator * options_.lr);
}

}  // namespace flashgrad::optim
