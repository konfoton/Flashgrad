#pragma once

#include <cstdint>

#include "flashgrad/core/tensor.h"

// Tensor operations. Phase 1 implements them on the CPU (the reference backend); calling
// them on CUDA tensors throws until the CUDA kernels arrive in Phase 2.
//
// Binary ops broadcast their operands (NumPy rules) and require both to have the same
// dtype and device — there is no implicit type promotion. Scalars are converted to the
// tensor's dtype (so `int_tensor * 2.5` multiplies by 2).
namespace flashgrad {

// ---- Elementwise binary ---------------------------------------------------------------
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);      // integer division truncates toward zero
Tensor maximum(const Tensor& a, const Tensor& b);  // NaN-propagating
Tensor minimum(const Tensor& a, const Tensor& b);  // NaN-propagating
Tensor pow(const Tensor& a, const Tensor& b);      // floating point only

Tensor add(const Tensor& a, double b);
Tensor sub(const Tensor& a, double b);
Tensor mul(const Tensor& a, double b);
Tensor div(const Tensor& a, double b);
Tensor pow(const Tensor& a, double b);

// ---- Comparisons (result dtype is bool) -----------------------------------------------
Tensor eq(const Tensor& a, const Tensor& b);
Tensor ne(const Tensor& a, const Tensor& b);
Tensor lt(const Tensor& a, const Tensor& b);
Tensor le(const Tensor& a, const Tensor& b);
Tensor gt(const Tensor& a, const Tensor& b);
Tensor ge(const Tensor& a, const Tensor& b);

// ---- Elementwise unary ----------------------------------------------------------------
Tensor neg(const Tensor& a);
Tensor abs(const Tensor& a);
Tensor relu(const Tensor& a);
Tensor exp(const Tensor& a);  // floating point only (as are the ones below)
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor sigmoid(const Tensor& a);

// ---- Reductions -----------------------------------------------------------------------
// Without `dim` they reduce over all elements and return a 0-dimensional tensor.
// sum/max/min keep the dtype, mean needs a floating-point tensor, argmax/argmin return int64
// (for the full reduction: the index into the flattened tensor). Floating-point sums are
// accumulated in double. max/min/argmax/argmin of zero elements throw; NaN propagates.
Tensor sum(const Tensor& a);
Tensor sum(const Tensor& a, std::int64_t dim, bool keepdim = false);
Tensor mean(const Tensor& a);
Tensor mean(const Tensor& a, std::int64_t dim, bool keepdim = false);
Tensor max(const Tensor& a);
Tensor max(const Tensor& a, std::int64_t dim, bool keepdim = false);
Tensor min(const Tensor& a);
Tensor min(const Tensor& a, std::int64_t dim, bool keepdim = false);
Tensor argmax(const Tensor& a);
Tensor argmax(const Tensor& a, std::int64_t dim, bool keepdim = false);
Tensor argmin(const Tensor& a);
Tensor argmin(const Tensor& a, std::int64_t dim, bool keepdim = false);

// ---- Linear algebra -------------------------------------------------------------------
// [..., m, k] x [..., k, n] -> [..., m, n]. Both inputs need at least 2 dimensions; leading
// (batch) dimensions broadcast.
Tensor matmul(const Tensor& a, const Tensor& b);

// ---- Normalization --------------------------------------------------------------------
// Numerically stable (subtracts the maximum); floating point only.
Tensor softmax(const Tensor& a, std::int64_t dim);
Tensor log_softmax(const Tensor& a, std::int64_t dim);

// ---- Comparison helpers (for tests) ---------------------------------------------------
// Same shape and |a - b| <= atol + rtol * |b| elementwise (computed in double on the host).
// NaNs are never close; infinities are close only to the same infinity.
bool allclose(const Tensor& a, const Tensor& b, double rtol = 1e-5, double atol = 1e-8);
// Same shape, dtype and exactly equal values.
bool equal(const Tensor& a, const Tensor& b);

// ---- Operators ------------------------------------------------------------------------
Tensor operator-(const Tensor& a);
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, double b);
Tensor operator-(const Tensor& a, double b);
Tensor operator*(const Tensor& a, double b);
Tensor operator/(const Tensor& a, double b);
Tensor operator+(double a, const Tensor& b);
Tensor operator-(double a, const Tensor& b);
Tensor operator*(double a, const Tensor& b);
Tensor operator/(double a, const Tensor& b);
Tensor& operator+=(Tensor& a, const Tensor& b);
Tensor& operator-=(Tensor& a, const Tensor& b);
Tensor& operator*=(Tensor& a, const Tensor& b);
Tensor& operator/=(Tensor& a, const Tensor& b);
Tensor& operator+=(Tensor& a, double b);
Tensor& operator-=(Tensor& a, double b);
Tensor& operator*=(Tensor& a, double b);
Tensor& operator/=(Tensor& a, double b);

}  // namespace flashgrad
