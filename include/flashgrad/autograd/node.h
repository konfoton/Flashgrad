#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "flashgrad/core/shape.h"
#include "flashgrad/core/tensor.h"

// The autograd graph. Every op that gets an input requiring grad (with grad mode on) gives
// its output a Node: the function that turns the output's gradient into the inputs'
// gradients, plus one Edge per input saying where each of those gradients goes next.
//
//   loss = sum(x * w)      loss --grad_fn--> sum --> mul --> AccumulateGrad(x)
//                                                        \-> AccumulateGrad(w)
//
// Ownership only points from outputs towards inputs, so the graph lives as long as the
// tensors that were computed from it.
namespace flashgrad::autograd {

class Node;

// Where the gradient for one input of a node goes: the input's grad_fn, or the
// AccumulateGrad node of a leaf. `node` is null when the input doesn't require grad.
// `shape` is the input's shape, so a gradient of a broadcast result can be summed back.
struct Edge {
  std::shared_ptr<Node> node;
  Shape shape;
};

class Node {
 public:
  // Gets the gradient of the op's output and returns one gradient per input, in the order of
  // the edges. An undefined tensor stands for zero. A gradient may have the broadcast shape
  // of the op's output; the engine sums it to the input's shape.
  using BackwardFn = std::function<std::vector<Tensor>(const Tensor& grad)>;

  // A persistent node (AccumulateGrad) survives release().
  Node(std::string name, std::vector<Edge> next, BackwardFn backward, bool persistent = false);

  const std::string& name() const noexcept { return name_; }
  const std::vector<Edge>& next() const noexcept { return next_; }

  // Runs the backward function. Throws once the node has been released.
  std::vector<Tensor> apply(const Tensor& grad);

  // Drops the backward function and the tensors it saved, after a backward pass that didn't
  // retain the graph. The edges stay, so the graph can still be walked.
  void release();
  bool released() const noexcept { return !backward_; }

 private:
  std::string name_;
  std::vector<Edge> next_;
  BackwardFn backward_;
  bool persistent_;
};

}  // namespace flashgrad::autograd
