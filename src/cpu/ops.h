#pragma once

#include "core/op_kinds.h"
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

// out, a, b have the same shape.
void binary(BinaryOp op, const Tensor& out, const Tensor& a, const Tensor& b);
// out, a, b have the same shape; out[i] is 1 where the comparison holds, else 0.
void compare(CompareOp op, const Tensor& out, const Tensor& a, const Tensor& b);
// out and a have the same shape.
void unary(UnaryOp op, const Tensor& out, const Tensor& a);

// in.shape() == out.shape() ++ reduced dims: reduces over the trailing in.dim() - out.dim()
// dimensions.
void reduce(ReduceOp op, const Tensor& out, const Tensor& in);
// Same layout as reduce; out holds the row-major index into the reduced dims.
void arg_reduce(ArgReduceOp op, const Tensor& out, const Tensor& in);

// a: [B..., m, k], b: [B..., k, n], out: [B..., m, n] with identical batch dims B.
void matmul(const Tensor& out, const Tensor& a, const Tensor& b);

// Softmax (or log-softmax) over the last dimension; out and in have the same shape.
void softmax(const Tensor& out, const Tensor& in, bool log_space);

}  // namespace flashgrad::cpu
