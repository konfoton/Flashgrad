#pragma once

#include <initializer_list>
#include <memory>

#include "flashgrad/autograd/node.h"
#include "flashgrad/core/tensor.h"

// Recording ops into the autograd graph. An op computes its output as usual, then:
//
//   if (autograd::detail::needs_grad({a, b})) {
//     autograd::detail::record(out, "mul", {a, b}, [a = a.detach(), b = b.detach()](
//                                                      const Tensor& g) {
//       return std::vector<Tensor>{g * b, g * a};
//     });
//   }
//
// Backward functions run with grad mode off. Tensors they save must be detached: a saved
// output would otherwise own its own grad_fn, a reference cycle.
namespace flashgrad::autograd::detail {

// True when grad mode is on and any input requires grad.
bool needs_grad(std::initializer_list<Tensor> inputs);

// Makes `out`, a tensor without autograd history, the output of a new node with one edge
// per input.
void record(const Tensor& out, const char* name, std::initializer_list<Tensor> inputs,
            Node::BackwardFn backward);

// Where a gradient for `t` goes: its grad_fn, or the AccumulateGrad node of a leaf.
std::shared_ptr<Node> gradient_edge(const Tensor& t);

// Runs the backward pass from `root` (see Tensor::backward).
void run_backward(const Tensor& root, const Tensor& gradient, bool retain_graph);

}  // namespace flashgrad::autograd::detail
