#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>

#include "cuda/memory.h"
#include "cuda/indexing.cuh"
#include "cuda/ops.h"
#include "flashgrad/core/error.h"
#include "flashgrad/cuda/check.h"

// Elementwise fill, map (unary) and zip (binary) kernels.
// One functor per op, one templated kernel per arity, operands addressed through strides.
namespace flashgrad::cuda::detail {

namespace {

// ---- Unary functors -------------------------------------------------------------------
struct NegFn {
  __device__ float operator()(float x) const { return -x; }
};
struct ReluFn {
  __device__ float operator()(float x) const { return x > 0.0f ? x : 0.0f; }
};
struct ExpFn {
  __device__ float operator()(float x) const { return expf(x); }
};
struct LogFn {
  __device__ float operator()(float x) const { return logf(x); }
};
struct SigmoidFn {
  // Never evaluates exp of a large positive number, so it can't overflow.
  __device__ float operator()(float x) const {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
  }
};
struct ReciprocalFn {
  __device__ float operator()(float x) const { return 1.0f / x; }
};

// ---- Binary functors ------------------------------------------------------------------
struct AddFn {
  __device__ float operator()(float a, float b) const { return a + b; }
};
struct SubFn {
  __device__ float operator()(float a, float b) const { return a - b; }
};
struct MulFn {
  __device__ float operator()(float a, float b) const { return a * b; }
};
struct DivFn {
  __device__ float operator()(float a, float b) const { return a / b; }
};
struct MaximumFn {
  // NaN in either operand gives NaN (a NaN `b` fails `a > b` and is returned).
  __device__ float operator()(float a, float b) const { return (a > b || a != a) ? a : b; }
};
struct LtFn {
  __device__ float operator()(float a, float b) const { return a < b ? 1.0f : 0.0f; }
};
struct EqFn {
  __device__ float operator()(float a, float b) const { return a == b ? 1.0f : 0.0f; }
};

// ---- Kernels --------------------------------------------------------------------------
// `calc` maps a linear index to the offsets of out and the inputs (IdentityCalc when all are
// contiguous, OffsetCalc otherwise).
// The value is a kernel argument, so filling needs no host -> device copy.
template <typename Calc>
__global__ void fill_kernel(float* __restrict__ out, float value, Calc calc, std::int64_t n) {
  const std::int64_t step = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += step) {
    std::int64_t off[1];
    calc.offsets(i, off);
    out[off[0]] = value;
  }
}

template <typename F, typename Calc>
__global__ void map_kernel(F f, float* __restrict__ out, const float* __restrict__ a, Calc calc,
                           std::int64_t n) {
  const std::int64_t step = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += step) {
    std::int64_t off[2];
    calc.offsets(i, off);
    out[off[0]] = f(a[off[1]]);
  }
}

template <typename F, typename Calc>
__global__ void zip_kernel(F f, float* __restrict__ out, const float* __restrict__ a,
                           const float* __restrict__ b, Calc calc, std::int64_t n) {
  const std::int64_t step = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += step) {
    std::int64_t off[3];
    calc.offsets(i, off);
    out[off[0]] = f(a[off[1]], b[off[2]]);
  }
}

template <typename F>
void launch_map(F f, const Tensor& out, const Tensor& a, Stream stream) {
  const std::int64_t n = out.numel();
  constexpr std::uint32_t thread_count = 256;
  constexpr std::uint32_t shm_bytes = 0;
  const auto num_blocks = static_cast<std::uint32_t>((n + thread_count - 1) / thread_count);
  if (out.is_contiguous() && a.is_contiguous()) {
    map_kernel<<<num_blocks, thread_count, shm_bytes, stream>>>(f, out.data_ptr(), a.data_ptr(),
                                                                IdentityCalc<2>{}, n);
  } else {
    const auto calc = make_offset_calc<2>(out.shape(), {&out.strides(), &a.strides()});
    map_kernel<<<num_blocks, thread_count, shm_bytes, stream>>>(f, out.data_ptr(), a.data_ptr(),
                                                                calc, n);
  }
  FG_CUDA_KERNEL_CHECK();
}

template <typename F>
void launch_zip(F f, const Tensor& out, const Tensor& a, const Tensor& b, Stream stream) {
  const std::int64_t n = out.numel();
  constexpr std::uint32_t thread_count = 256;
  constexpr std::uint32_t shm_bytes = 0;
  const auto num_blocks = static_cast<std::uint32_t>((n + thread_count - 1) / thread_count);
  if (out.is_contiguous() && a.is_contiguous() && b.is_contiguous()) {
    zip_kernel<<<num_blocks, thread_count, shm_bytes, stream>>>(
        f, out.data_ptr(), a.data_ptr(), b.data_ptr(), IdentityCalc<3>{}, n);
  } else {
    const auto calc =
        make_offset_calc<3>(out.shape(), {&out.strides(), &a.strides(), &b.strides()});
    zip_kernel<<<num_blocks, thread_count, shm_bytes, stream>>>(f, out.data_ptr(), a.data_ptr(),
                                                                b.data_ptr(), calc, n);
  }
  FG_CUDA_KERNEL_CHECK();
}

}  // namespace

void fill(const Tensor& out, float value, Stream stream) {
  const std::int64_t n = out.numel();
  if (n == 0) return;
  DeviceGuard guard(out.device().index());
  constexpr std::uint32_t thread_count = 256;
  constexpr std::uint32_t shm_bytes = 0;
  const auto num_blocks =
      static_cast<std::uint32_t>(std::min((n + thread_count - 1) / thread_count, kMaxBlocks));
  if (out.is_contiguous()) {
    fill_kernel<<<num_blocks, thread_count, shm_bytes, stream>>>(out.data_ptr(), value,
                                                                 IdentityCalc<1>{}, n);
  } else {
    const auto calc = make_offset_calc<1>(out.shape(), {&out.strides()});
    fill_kernel<<<num_blocks, thread_count, shm_bytes, stream>>>(out.data_ptr(), value, calc, n);
  }
  FG_CUDA_KERNEL_CHECK();
}

void unary(UnaryOp op, const Tensor& out, const Tensor& a, Stream stream) {
  if (out.numel() == 0) return;
  DeviceGuard guard(out.device().index());
  switch (op) {
    case UnaryOp::Neg: return launch_map(NegFn{}, out, a, stream);
    case UnaryOp::Relu: return launch_map(ReluFn{}, out, a, stream);
    case UnaryOp::Exp: return launch_map(ExpFn{}, out, a, stream);
    case UnaryOp::Log: return launch_map(LogFn{}, out, a, stream);
    case UnaryOp::Sigmoid: return launch_map(SigmoidFn{}, out, a, stream);
    case UnaryOp::Reciprocal: return launch_map(ReciprocalFn{}, out, a, stream);
  }
  FG_THROW("unary: unknown op ", static_cast<int>(op));
}

void binary(BinaryOp op, const Tensor& out, const Tensor& a, const Tensor& b, Stream stream) {
  if (out.numel() == 0) return;
  DeviceGuard guard(out.device().index());
  switch (op) {
    case BinaryOp::Add: return launch_zip(AddFn{}, out, a, b, stream);
    case BinaryOp::Sub: return launch_zip(SubFn{}, out, a, b, stream);
    case BinaryOp::Mul: return launch_zip(MulFn{}, out, a, b, stream);
    case BinaryOp::Div: return launch_zip(DivFn{}, out, a, b, stream);
    case BinaryOp::Maximum: return launch_zip(MaximumFn{}, out, a, b, stream);
    case BinaryOp::Lt: return launch_zip(LtFn{}, out, a, b, stream);
    case BinaryOp::Eq: return launch_zip(EqFn{}, out, a, b, stream);
  }
  FG_THROW("binary: unknown op ", static_cast<int>(op));
}

}  // namespace flashgrad::cuda::detail
