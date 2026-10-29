#pragma once

#include <memory>

#include "flashgrad/autograd/node.h"
#include "flashgrad/core/tensor.h"

namespace flashgrad::autograd {

// Autograd state of a tensor, allocated only for tensors that take part in autograd.
struct AutogradMeta {
  bool requires_grad = false;          // set on leaves; outputs of recorded ops always require grad
  Tensor grad;                         // leaves: accumulated by backward
  std::shared_ptr<Node> grad_fn;       // outputs of recorded ops; null for leaves
  std::shared_ptr<Node> accumulator;   // leaves: the AccumulateGrad node, created on first use
};

}  // namespace flashgrad::autograd
