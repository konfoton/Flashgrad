#pragma once

#include "flashgrad/core/tensor.h"

// Host-side copy and fill for CPU tensors (creating data, printing, staging transfers).
// Flashgrad has no CPU compute ops: all math runs as CUDA kernels.
//
// Preconditions (checked by the core layer, not here): all tensors are on the CPU and are
// float32; any strides are accepted.
namespace flashgrad::cpu {

// dst and src have the same shape.
void copy(const Tensor& dst, const Tensor& src);
void fill(const Tensor& out, double value);

}  // namespace flashgrad::cpu
