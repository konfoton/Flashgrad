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
Tensor mul(const Tensor& a, const Tensor& b);
Tensor maximum(const Tensor& a, const Tensor& b);
// Comparisons: 1.0 where true, 0.0 where false.
Tensor lt(const Tensor& a, const Tensor& b);
Tensor eq(const Tensor& a, const Tensor& b);

// ---- Reductions over one dimension ----------------------------------------------------
// Accumulated in float32. The reduced dimension is removed unless `keepdim` is true.
// Over an empty dimension sum gives 0 and prod gives 1; max throws. max propagates NaN.
Tensor sum(const Tensor& a, std::int64_t dim, bool keepdim = false);
Tensor prod(const Tensor& a, std::int64_t dim, bool keepdim = false);
Tensor max(const Tensor& a, std::int64_t dim, bool keepdim = false);

}  // namespace flashgrad
