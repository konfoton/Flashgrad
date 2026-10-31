#pragma once

#include <cstdint>

#include "flashgrad/core/tensor.h"

// Same type as cudaStream_t, declared here so core code doesn't depend on cuda_runtime.h.
struct CUstream_st;

// Kernel launchers behind the public ops. Kept free of CUDA headers like cuda/memory.h.
//
// Preconditions: every tensor is a float32 CUDA tensor
// on the same device with at most kMaxDims dimensions; `out` is freshly allocated, so it does
// not overlap any input. Inputs may have any strides, including 0 for expanded dims.
namespace flashgrad::cuda::detail {

using Stream = CUstream_st*;  // nullptr is the default stream

inline constexpr std::int64_t kMaxDims = 8;
inline constexpr std::int64_t kMaxBlocks = 0x7fffffff;

enum class UnaryOp { Identity, Neg, Relu, Exp, Log, Sigmoid, Reciprocal };  // Identity copies
enum class BinaryOp { Add, Sub, Mul, Div, Maximum, Lt, Eq };
enum class ReduceOp { Sum, Prod, Max };

// Sets every element of `out` to `value`. Unlike the other launchers, `out` may be any
// existing tensor or view, including one with expanded (stride 0) dims, and may have more
// than kMaxDims dimensions if it is contiguous.
void fill(const Tensor& out, float value, Stream stream = nullptr);

// out = a, element by element; `a` has the shape of `out` (broadcast beforehand with expand).
// `out` may be any existing view whose elements don't overlap each other or `a`.
void copy(const Tensor& out, const Tensor& a, Stream stream = nullptr);

// out = op(a); `a` has the shape of `out`.
void unary(UnaryOp op, const Tensor& out, const Tensor& a, Stream stream = nullptr);

// out = op(a, b); `a` and `b` have the shape of `out` (broadcast beforehand with expand).
void binary(BinaryOp op, const Tensor& out, const Tensor& a, const Tensor& b,
            Stream stream = nullptr);

// Reduces `a` along `dim` into `out`, which has the shape of `a` with size 1 at `dim`.
void reduce(ReduceOp op, const Tensor& out, const Tensor& a, std::int64_t dim,
            Stream stream = nullptr);

// Reduces all elements of `a` into the 0-dim `out`; an empty `a` gives the op's identity.
void reduce_all(ReduceOp op, const Tensor& out, const Tensor& a, Stream stream = nullptr);

// out = a @ b for 2-D `a` [M, K] and `b` [K, N]; `out` is [M, N]. K may be 0 (out is zeroed).
void matmul(const Tensor& out, const Tensor& a, const Tensor& b, Stream stream = nullptr);

// out = log_softmax(a) along `dim`; `out` has the shape of `a`.
void log_softmax(const Tensor& out, const Tensor& a, std::int64_t dim, Stream stream = nullptr);

}  // namespace flashgrad::cuda::detail
