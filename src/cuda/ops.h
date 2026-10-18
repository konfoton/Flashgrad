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

enum class UnaryOp { Neg, Relu, Exp, Log, Sigmoid, Reciprocal };
enum class BinaryOp { Add, Sub, Mul, Div, Maximum, Lt, Eq };
enum class ReduceOp { Sum, Prod, Max };

// out = op(a); `a` has the shape of `out`.
void unary(UnaryOp op, const Tensor& out, const Tensor& a, Stream stream = nullptr);

// out = op(a, b); `a` and `b` have the shape of `out` (broadcast beforehand with expand).
void binary(BinaryOp op, const Tensor& out, const Tensor& a, const Tensor& b,
            Stream stream = nullptr);

// Reduces `a` along `dim` into `out`, which has the shape of `a` with size 1 at `dim`.
void reduce(ReduceOp op, const Tensor& out, const Tensor& a, std::int64_t dim,
            Stream stream = nullptr);

}  // namespace flashgrad::cuda::detail
