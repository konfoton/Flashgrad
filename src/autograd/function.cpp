#include "autograd/function.h"

#include <memory>
#include <utility>
#include <vector>

#include "autograd/meta.h"
#include "flashgrad/autograd/grad_mode.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"

namespace flashgrad::autograd::detail {

namespace {

// Adds `grad` to the gradient of `leaf`. The first gradient is copied, so the leaf never
// shares memory with another tensor's gradient (add's backward hands the same tensor to both
// inputs) and in-place changes to grad() stay local.
void accumulate(AutogradMeta& leaf, const Tensor& grad) {
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
    next.push_back({t.requires_grad() ? gradient_edge(t) : nullptr, t.shape()});
  }
  AutogradMeta& meta = out.impl()->ensure_autograd_meta();
  FG_CHECK(meta.grad_fn == nullptr && !meta.requires_grad, name,
           ": internal error, the output already has autograd history");
  meta.grad_fn = std::make_shared<Node>(name, std::move(next), std::move(backward));
}

std::shared_ptr<Node> gradient_edge(const Tensor& t) {
  if (std::shared_ptr<Node> fn = t.grad_fn()) return fn;
  AutogradMeta& meta = t.impl()->ensure_autograd_meta();
  if (!meta.accumulator) {
    // A weak reference: the leaf owns its accumulator, not the other way round.
    const std::weak_ptr<TensorImpl> leaf = t.impl();
    meta.accumulator = std::make_shared<Node>(
        "AccumulateGrad", std::vector<Edge>{},
        [leaf](const Tensor& grad) {
          if (const std::shared_ptr<TensorImpl> impl = leaf.lock()) {
            accumulate(*impl->autograd_meta(), grad);
          }
          return std::vector<Tensor>{};
        },
        /*persistent=*/true);
  }
  return meta.accumulator;
}

}  // namespace flashgrad::autograd::detail
