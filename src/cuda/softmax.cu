#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "cuda/block_reduce.cuh"
#include "cuda/indexing.cuh"
#include "cuda/memory.h"
#include "cuda/ops.h"
#include "flashgrad/core/shape.h"
#include "flashgrad/cuda/check.h"

// log_softmax along one dimension. One block per row, three passes over the row:
// the max, the sum of exp(x - max), then out = (x - max) - log(sum). Subtracting the max
// keeps exp from overflowing.
namespace flashgrad::cuda::detail {

namespace {

// calc covers the rows shape (size 1 at `dim`) with the strides of out and a, so it yields
// the start of the row in both.
__global__ void log_softmax_kernel(float* __restrict__ out, const float* __restrict__ a,
                                   OffsetCalc<2> calc, std::int64_t n_rows,
                                   std::int64_t row_size, std::int64_t out_stride,
                                   std::int64_t a_stride) {
  extern __shared__ float warp_partials[];

  for (std::int64_t r = blockIdx.x; r < n_rows; r += gridDim.x) {
    std::int64_t off[2];
    calc.offsets(r, off);
    float* y = out + off[0];
    const float* x = a + off[1];

    float max = MaxFn::kIdentity;
    for (std::int64_t i = threadIdx.x; i < row_size; i += blockDim.x) {
      max = MaxFn{}(max, x[i * a_stride]);
    }
    max = block_reduce(MaxFn{}, max, warp_partials);

    float sum = 0.0f;
    for (std::int64_t i = threadIdx.x; i < row_size; i += blockDim.x) {
      sum += expf(x[i * a_stride] - max);
    }
    const float log_sum = logf(block_reduce(SumFn{}, sum, warp_partials));

    for (std::int64_t i = threadIdx.x; i < row_size; i += blockDim.x) {
      y[i * out_stride] = (x[i * a_stride] - max) - log_sum;
    }
  }
}

}  // namespace

void log_softmax(const Tensor& out, const Tensor& a, std::int64_t dim, Stream stream) {
  if (out.numel() == 0) return;
  DeviceGuard guard(out.device().index());
  Shape rows = a.shape();
  rows[dim] = 1;
  const std::int64_t n_rows = out.numel() / a.size(dim);
  const std::int64_t row_size = a.size(dim);
  const std::uint32_t thread_count = row_block_threads(row_size);
  const std::uint32_t shm_bytes = thread_count / kWarpSize * sizeof(float);
  const auto num_blocks = static_cast<std::uint32_t>(std::min(n_rows, kMaxBlocks));
  const auto calc = make_offset_calc<2>(rows, {&out.strides(), &a.strides()});
  log_softmax_kernel<<<num_blocks, thread_count, shm_bytes, stream>>>(
      out.data_ptr(), a.data_ptr(), calc, n_rows, row_size, out.stride(dim), a.stride(dim));
  FG_CUDA_KERNEL_CHECK();
}

}  // namespace flashgrad::cuda::detail
