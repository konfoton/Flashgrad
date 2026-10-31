#include "flashgrad/autograd/node.h"

#include <utility>

#include "flashgrad/core/error.h"

namespace flashgrad::autograd {

Node::Node(std::string name, std::vector<Edge> next, BackwardFn backward, bool persistent)
    : name_(std::move(name)),
      next_(std::move(next)),
      backward_(std::move(backward)),
      persistent_(persistent) {
  FG_CHECK(backward_ != nullptr, "Node: ", name_, " needs a backward function");
}

std::vector<Tensor> Node::apply(const Tensor& grad) {
  FG_CHECK(!released(), "backward: ", name_,
           " was already used by a backward pass that freed the graph; pass retain_graph = true "
           "to the first backward to go through the graph again");
  std::vector<Tensor> grads = backward_(grad);
  FG_CHECK(persistent_ || grads.size() == next_.size(), "backward: ", name_, " returned ",
           grads.size(), " gradients for ", next_.size(), " inputs");
  return grads;
}

void Node::release() {
  if (!persistent_) backward_ = nullptr;
}

}  // namespace flashgrad::autograd
