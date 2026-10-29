// The backward pass: walks the graph from the root in topological order, so a node runs
// only after every node that uses its output has sent its gradient.
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "autograd/function.h"
#include "core/checks.h"
#include "flashgrad/autograd/grad_mode.h"
#include "flashgrad/autograd/node.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/shape.h"

namespace flashgrad {

namespace autograd::detail {

namespace {

// A gradient may have the broadcast shape of the op's output; sum it back to the input's.
Tensor to_input_shape(const Tensor& grad, const Shape& shape, const Node& node) {
  if (grad.shape() == shape) return grad;
  FG_CHECK(broadcast_shapes(shape, grad.shape()) == grad.shape(), "backward: ", node.name(),
           " returned a gradient ", describe(grad), " for an input of shape ", to_string(shape));
  return unbroadcast(grad, shape);
}

}  // namespace

void run_backward(const Tensor& root, const Tensor& gradient, bool retain_graph) {
  flashgrad::detail::check_defined("backward", root);
  FG_CHECK(root.requires_grad(), "backward: ", describe(root),
           " doesn't require grad: no input requires grad, or it was computed with grad mode off");
  Tensor seed = gradient;
  if (seed.defined()) {
    FG_CHECK(seed.shape() == root.shape() && seed.device() == root.device(),
             "backward: gradient ", describe(seed), " doesn't match the output ", describe(root));
  } else {
    FG_CHECK(root.numel() == 1, "backward: pass a gradient for the non-scalar output ",
             describe(root));
    seed = ones_like(root);
  }

  const NoGradGuard no_grad;
  const std::shared_ptr<Node> start = gradient_edge(root);

  // How many edges lead into each node from the part of the graph reachable from the root.
  std::unordered_map<Node*, int> pending;
  std::unordered_set<Node*> seen{start.get()};
  std::vector<Node*> stack{start.get()};
  while (!stack.empty()) {
    Node* node = stack.back();
    stack.pop_back();
    for (const Edge& edge : node->next()) {
      if (!edge.node) continue;
      ++pending[edge.node.get()];
      if (seen.insert(edge.node.get()).second) stack.push_back(edge.node.get());
    }
  }

  // Kahn's algorithm. grads[node] sums what the node's consumers sent so far; a node is ready
  // once all of them have run.
  std::unordered_map<Node*, Tensor> grads{{start.get(), seed}};
  std::vector<Node*> ready{start.get()};
  while (!ready.empty()) {
    Node* node = ready.back();
    ready.pop_back();
    const auto it = grads.find(node);
    const Tensor grad = it == grads.end() ? Tensor() : it->second;
    if (it != grads.end()) grads.erase(it);
    // A node whose gradient is zero (undefined) sends nothing, but still counts as done.
    const std::vector<Tensor> input_grads = grad.defined() ? node->apply(grad)
                                                           : std::vector<Tensor>{};
    for (std::size_t i = 0; i < node->next().size(); ++i) {
      const Edge& edge = node->next()[i];
      if (!edge.node) continue;
      if (i < input_grads.size() && input_grads[i].defined()) {
        const Tensor g = to_input_shape(input_grads[i], edge.shape, *node);
        Tensor& total = grads[edge.node.get()];
        total = total.defined() ? total + g : g;
      }
      if (--pending[edge.node.get()] == 0) ready.push_back(edge.node.get());
    }
  }

  if (!retain_graph) {
    for (Node* node : seen) node->release();
  }
}

}  // namespace autograd::detail

void Tensor::backward(const Tensor& gradient, bool retain_graph) const {
  autograd::detail::run_backward(*this, gradient, retain_graph);
}

}  // namespace flashgrad
