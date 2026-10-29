#include "autograd/function.h"

#include <memory>
#include <utility>
#include <vector>

#include "autograd/grad_state.h"
#include "flashgrad/autograd/grad_mode.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"

namespace flashgrad::autograd::detail {

namespace {

// Adds `grad` to the gradient of `leaf`. The first gradient is copied, so the leaf never
// shares memory with another tensor's gradient (add's backward hands the same tensor to both
// inputs) and in-place changes to grad() stay local.
void accumulate(GradState& leaf, const Tensor& grad) {
  leaf.grad = leaf.grad.defined() ? leaf.grad + grad : grad.clone();
}

}  // namespace

bool needs_grad(std::initializer_list<Tensor> inputs) {
  if (!is_grad_enabled()) return false;
  for (const Tensor& t : inputs) {
    if (t.defined() && t.requires_grad()) return true;
  }
  return false;
}

void record(const Tensor& out, const char* name, std::initializer_list<Tensor> inputs,
            Node::BackwardFn backward) {
  std::vector<Edge> next;
  next.reserve(inputs.size());
  for (const Tensor& t : inputs) {
    next.push_back({t.requires_grad() ? grad_target(t) : nullptr, t.shape()});
  }
  GradState& state = out.impl()->ensure_grad_state();
  FG_CHECK(state.grad_fn == nullptr && !state.requires_grad, name,
           ": internal error, the output already has autograd history");
  state.grad_fn = std::make_shared<Node>(name, std::move(next), std::move(backward));
}

std::shared_ptr<Node> grad_target(const Tensor& t) {
  if (std::shared_ptr<Node> fn = t.grad_fn()) return fn;
  GradState& state = t.impl()->ensure_grad_state();
  if (!state.leaf_node) {
    // A weak reference: the leaf owns this node, not the other way round.
    const std::weak_ptr<TensorView> leaf = t.impl();
    state.leaf_node = std::make_shared<Node>(
        "LeafGrad", std::vector<Edge>{},
        [leaf](const Tensor& grad) {
          if (const std::shared_ptr<TensorView> impl = leaf.lock()) {
            accumulate(*impl->grad_state(), grad);
          }
          return std::vector<Tensor>{};
        },
        /*persistent=*/true);
  }
  return state.leaf_node;
}

}  // namespace flashgrad::autograd::detail
