#pragma once

#include <cstdint>

#include "flashgrad/core/tensor.h"

// Tensor math ops. Every op runs as a CUDA kernel: inputs must be CUDA tensors (all on the
// same device) and the result is a new contiguous float32 tensor on that device. Inputs may
// have any strides (views, transposes, expanded dims) and at most 8 dimensions.
namespace flashgrad {

// ---- Elementwise unary ----------------------------------------------------------------
Tensor neg(const Tensor& a);
Tensor relu(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sigmoid(const Tensor& a);  // numerically stable for large |x|
Tensor reciprocal(const Tensor& a);

// ---- Elementwise binary (operands broadcast, NumPy rules) -----------------------------
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor maximum(const Tensor& a, const Tensor& b);
// Comparisons: 1.0 where true, 0.0 where false. Not differentiable: the results never
// require grad, so they can serve as masks.
Tensor lt(const Tensor& a, const Tensor& b);
Tensor eq(const Tensor& a, const Tensor& b);

// ---- Reductions over one dimension ----------------------------------------------------
// Accumulated in float32. The reduced dimension is removed unless `keepdim` is true.
// Over an empty dimension sum gives 0 and prod gives 1; max throws. max propagates NaN.
Tensor sum(const Tensor& a, std::int64_t dim, bool keepdim = false);
Tensor prod(const Tensor& a, std::int64_t dim, bool keepdim = false);
Tensor max(const Tensor& a, std::int64_t dim, bool keepdim = false);
// sum / size of the dimension; an empty dimension gives NaN.
Tensor mean(const Tensor& a, std::int64_t dim, bool keepdim = false);

// ---- Reductions over all elements -----------------------------------------------------
// 0-dim results, accumulated in float32. An empty tensor sums to 0; its mean is NaN.
Tensor sum(const Tensor& a);
Tensor mean(const Tensor& a);

// Sums `a` down to `shape`, the inverse of broadcasting: `shape` must broadcast to
// a.shape(). Leading dims are summed away and dims of size 1 in `shape` are summed with
// keepdim, e.g. [4, 2, 3] -> [2, 1]. Returns `a` itself when the shapes are equal.
Tensor unbroadcast(const Tensor& a, const Shape& shape);

// ---- Softmax -------------------------------------------------------------------------
// log(softmax(a)) along `dim`, computed as (a - max) - log(sum(exp(a - max))) so large
// inputs don't overflow. A NaN in a row makes the whole row NaN.
Tensor log_softmax(const Tensor& a, std::int64_t dim);

// ---- Matrix multiplication -----------------------------------------------------------
// Product of matrices a [M, K] and b [K, N], giving [M, N]; accumulated in float32.
// Inputs may be strided views; ones that aren't contiguous with rows a multiple of 4 floats
// long are first copied into a zero-padded buffer.
Tensor matmul(const Tensor& a, const Tensor& b);

// ---- Operators ------------------------------------------------------------------------
// Shorthands for the arithmetic ops. A scalar operand becomes a 0-dim tensor on the other
// operand's device and broadcasts.
Tensor operator-(const Tensor& a);
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, float s);
Tensor operator-(const Tensor& a, float s);
Tensor operator*(const Tensor& a, float s);
Tensor operator/(const Tensor& a, float s);
Tensor operator+(float s, const Tensor& a);
Tensor operator-(float s, const Tensor& a);
Tensor operator*(float s, const Tensor& a);
Tensor operator/(float s, const Tensor& a);

}  // namespace flashgrad
