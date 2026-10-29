#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "cuda/block_reduce.cuh"
#include "cuda/memory.h"
#include "cuda/indexing.cuh"
#include "cuda/ops.h"
#include "flashgrad/core/error.h"
#include "flashgrad/cuda/check.h"

// Reductions. Along one dimension: one block per output element, a strided per-thread loop,
// then a block-wide reduction (block_reduce.cuh). Over all elements: two passes, first a
// partial result per block, then one block combines the partials.
namespace flashgrad::cuda::detail {

namespace {

// calc covers the output shape (size 1 at the reduced dim) with the strides of out and a, so
// it yields the output element and the start of its row in `a`.
template <typename F>
__global__ void reduce_kernel(F f, float* __restrict__ out, const float* __restrict__ a,
                              OffsetCalc<2> calc, std::int64_t n_out, std::int64_t reduce_size,
                              std::int64_t reduce_stride) {
  extern __shared__ float warp_partials[];

  for (std::int64_t o = blockIdx.x; o < n_out; o += gridDim.x) {
    std::int64_t off[2];
    calc.offsets(o, off);
    const float* row = a + off[1];

    float v = F::kIdentity;
    for (std::int64_t i = threadIdx.x; i < reduce_size; i += blockDim.x) {
      v = f(v, row[i * reduce_stride]);
    }
    v = block_reduce(f, v, warp_partials);
    if (threadIdx.x == 0) out[off[0]] = v;
  }
}

template <typename F>
void launch_reduce(F f, const Tensor& out, const Tensor& a, std::int64_t dim, Stream stream) {
  const std::int64_t n_out = out.numel();
  const std::int64_t reduce_size = a.size(dim);
  // Short rows don't need idle warps.
  const std::uint32_t thread_count = row_block_threads(reduce_size);
  const std::uint32_t shm_bytes = thread_count / kWarpSize * sizeof(float);
  const auto num_blocks = static_cast<std::uint32_t>(std::min(n_out, kMaxBlocks));
  const auto calc = make_offset_calc<2>(out.shape(), {&out.strides(), &a.strides()});
  reduce_kernel<<<num_blocks, thread_count, shm_bytes, stream>>>(
      f, out.data_ptr(), a.data_ptr(), calc, n_out, reduce_size, a.stride(dim));
  FG_CUDA_KERNEL_CHECK();
}

// Each block combines a grid-stride share of the n elements of `a` into out[blockIdx.x].
template <typename F, typename Calc>
__global__ void reduce_all_kernel(F f, float* __restrict__ out, const float* __restrict__ a,
                                  Calc calc, std::int64_t n) {
  __shared__ float warp_partials[kMaxRowThreads / kWarpSize];
  const std::int64_t step = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  float v = F::kIdentity;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += step) {
    std::int64_t off[1];
    calc.offsets(i, off);
    v = f(v, a[off[0]]);
  }
  v = block_reduce(f, v, warp_partials);
  if (threadIdx.x == 0) out[blockIdx.x] = v;
}

// Enough blocks to fill the GPU several times over; more would only add partials.
constexpr std::int64_t kMaxReduceAllBlocks = 1024;

template <typename F>
void launch_reduce_all(F f, const Tensor& out, const Tensor& a, Stream stream) {
  const std::int64_t n = a.numel();
  constexpr std::uint32_t thread_count = kMaxRowThreads;
  constexpr std::uint32_t shm_bytes = 0;
  const auto num_blocks = static_cast<std::uint32_t>(
      std::clamp<std::int64_t>((n + thread_count - 1) / thread_count, 1, kMaxReduceAllBlocks));
  // A single block writes the result directly; otherwise the partials go to a buffer.
  const Tensor partials =
      num_blocks == 1 ? out : empty({static_cast<std::int64_t>(num_blocks)}, out.options());
  if (a.is_contiguous()) {
    reduce_all_kernel<<<num_blocks, thread_count, shm_bytes, stream>>>(
        f, partials.data_ptr(), a.data_ptr(), IdentityCalc<1>{}, n);
  } else {
    const auto calc = make_offset_calc<1>(a.shape(), {&a.strides()});
    reduce_all_kernel<<<num_blocks, thread_count, shm_bytes, stream>>>(
        f, partials.data_ptr(), a.data_ptr(), calc, n);
  }
  FG_CUDA_KERNEL_CHECK();
  if (num_blocks == 1) return;
  reduce_all_kernel<<<1, thread_count, shm_bytes, stream>>>(
      f, out.data_ptr(), partials.data_ptr(), IdentityCalc<1>{}, num_blocks);
  FG_CUDA_KERNEL_CHECK();
}

}  // namespace

void reduce_all(ReduceOp op, const Tensor& out, const Tensor& a, Stream stream) {
  DeviceGuard guard(out.device().index());
  switch (op) {
    case ReduceOp::Sum: return launch_reduce_all(SumFn{}, out, a, stream);
    case ReduceOp::Prod: return launch_reduce_all(ProdFn{}, out, a, stream);
    case ReduceOp::Max: return launch_reduce_all(MaxFn{}, out, a, stream);
  }
  FG_THROW("reduce_all: unknown op ", static_cast<int>(op));
}

void reduce(ReduceOp op, const Tensor& out, const Tensor& a, std::int64_t dim, Stream stream) {
  if (out.numel() == 0) return;
  DeviceGuard guard(out.device().index());
  switch (op) {
    case ReduceOp::Sum: return launch_reduce(SumFn{}, out, a, dim, stream);
    case ReduceOp::Prod: return launch_reduce(ProdFn{}, out, a, dim, stream);
    case ReduceOp::Max: return launch_reduce(MaxFn{}, out, a, dim, stream);
  }
  FG_THROW("reduce: unknown op ", static_cast<int>(op));
}

}  // namespace flashgrad::cuda::detail
