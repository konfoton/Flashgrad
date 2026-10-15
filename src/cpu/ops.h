#pragma once

#include "flashgrad/core/tensor.h"

// CPU reference kernels. They are deliberately naive (strided loops, double accumulators)
// so they are obviously correct; the CUDA kernels are tested against them.
//
// Preconditions (checked by the core layer, not here): all tensors are on the CPU, `out` is
// allocated with the documented shape and dtype, and inputs are already broadcast to the
// shapes stated below. All tensors are float32; any strides are accepted.
namespace flashgrad::cpu {

// dst and src have the same shape.
void copy(const Tensor& dst, const Tensor& src);
void fill(const Tensor& out, double value);

}  // namespace flashgrad::cpu
