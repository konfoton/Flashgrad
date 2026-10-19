#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "cuda/block_reduce.cuh"
#include "cuda/memory.h"
#include "cuda/indexing.cuh"
#include "cuda/ops.h"
#include "flashgrad/core/error.h"
#include "flashgrad/cuda/check.h"

// Reduction along one dimension. One block per output element, a strided per-thread loop,
// then a block-wide reduction (block_reduce.cuh).
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

}  // namespace

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
