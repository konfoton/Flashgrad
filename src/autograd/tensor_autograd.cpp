// The autograd methods of Tensor.
#include <memory>

#include "autograd/grad_state.h"
#include "core/checks.h"
#include "flashgrad/autograd/node.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/tensor.h"

namespace flashgrad {

namespace {

const autograd::GradState* state_of(const Tensor& t) {
  detail::check_defined("autograd", t);
  return t.impl()->grad_state();
}

}  // namespace

bool Tensor::requires_grad() const {
  const autograd::GradState* state = state_of(*this);
  return state != nullptr && (state->requires_grad || state->grad_fn != nullptr);
}

const Tensor& Tensor::requires_grad_(bool requires_grad) const {
  FG_CHECK(is_leaf() || requires_grad, "requires_grad_: ", describe(*this),
           " was computed by ", grad_fn()->name(),
           "; only leaves can stop requiring grad (use detach() instead)");
  if (is_leaf()) impl()->ensure_grad_state().requires_grad = requires_grad;
  return *this;
}

bool Tensor::is_leaf() const { return grad_fn() == nullptr; }

std::shared_ptr<autograd::Node> Tensor::grad_fn() const {
  const autograd::GradState* state = state_of(*this);
  return state == nullptr ? nullptr : state->grad_fn;
}

Tensor Tensor::grad() const {
  const autograd::GradState* state = state_of(*this);
  return state == nullptr ? Tensor() : state->grad;
}

void Tensor::set_grad(const Tensor& grad) const {
  detail::check_defined("set_grad", *this);
  FG_CHECK(!grad.defined() || (grad.shape() == shape() && grad.device() == device()),
           "set_grad: gradient ", describe(grad), " doesn't match ", describe(*this));
  impl()->ensure_grad_state().grad = grad;
}

Tensor Tensor::detach() const {
  detail::check_defined("detach", *this);
  return Tensor(std::make_shared<TensorView>(storage(), shape(), strides(), offset(), dtype()));
}

}  // namespace flashgrad
