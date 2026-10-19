#include <cuda_runtime.h>

#include <cstdint>

#include "cuda/memory.h"
#include "cuda/ops.h"
#include "flashgrad/core/error.h"
#include "flashgrad/cuda/check.h"

// Matrix multiply with shared-memory tiling. Each block computes one kTile x kTile tile of
// the output, one element per thread. It walks along K in steps of kTile: the block loads a
// tile of `a` and a tile of `b` into shared memory, then every thread reads its row and column
// of the two tiles from there. Each element of `a` and `b` is read from global memory
// N / kTile and M / kTile times instead of N and M times.
namespace flashgrad::cuda::detail {

namespace {

constexpr int kTile = 16;  // kTile * kTile = 256 threads per block

// out[m, n] = sum_k a[m, k] * b[k, n]; `a` and `b` are addressed through their strides.
// Block `blockIdx.x` computes output tile (blockIdx.x / tiles_n, blockIdx.x % tiles_n).
__global__ void matmul_kernel(float* __restrict__ out, const float* __restrict__ a,
                              const float* __restrict__ b, std::int64_t m, std::int64_t n,
                              std::int64_t k, std::int64_t a_row_stride,
                              std::int64_t a_col_stride, std::int64_t b_row_stride,
                              std::int64_t b_col_stride) {
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];
  const int ty = static_cast<int>(threadIdx.y);
  const int tx = static_cast<int>(threadIdx.x);
  const std::int64_t tiles_n = (n + kTile - 1) / kTile;
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) / tiles_n * kTile + ty;
  const std::int64_t col = static_cast<std::int64_t>(blockIdx.x) % tiles_n * kTile + tx;

  float acc = 0.0f;
  for (std::int64_t k0 = 0; k0 < k; k0 += kTile) {
    // Out-of-range positions load 0, so they add nothing to the dot product.
    const std::int64_t a_col = k0 + tx;
    const std::int64_t b_row = k0 + ty;
    a_tile[ty][tx] = row < m && a_col < k ? a[row * a_row_stride + a_col * a_col_stride] : 0.0f;
    b_tile[ty][tx] = b_row < k && col < n ? b[b_row * b_row_stride + col * b_col_stride] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kTile; ++i) acc += a_tile[ty][i] * b_tile[i][tx];
    __syncthreads();  // the next step overwrites the tiles
  }
  if (row < m && col < n) out[row * n + col] = acc;
}

}  // namespace

void matmul(const Tensor& out, const Tensor& a, const Tensor& b, Stream stream) {
  if (out.numel() == 0) return;
  DeviceGuard guard(out.device().index());
  const std::int64_t m = a.size(0);
  const std::int64_t k = a.size(1);
  const std::int64_t n = b.size(1);
  const std::int64_t num_tiles = (m + kTile - 1) / kTile * ((n + kTile - 1) / kTile);
  // Needs m * n > 5e11 to trigger, far more than fits in GPU memory.
  FG_CHECK(num_tiles <= kMaxBlocks, "matmul: output [", m, ", ", n, "] has more tiles (",
           num_tiles, ") than the grid limit ", kMaxBlocks);
  const auto num_blocks = static_cast<std::uint32_t>(num_tiles);
  const dim3 thread_count(kTile, kTile);
  constexpr std::uint32_t shm_bytes = 0;  // the tiles are static shared memory
  matmul_kernel<<<num_blocks, thread_count, shm_bytes, stream>>>(
      out.data_ptr(), a.data_ptr(), b.data_ptr(), m, n, k, a.stride(0), a.stride(1),
      b.stride(0), b.stride(1));
  FG_CUDA_KERNEL_CHECK();
}

}  // namespace flashgrad::cuda::detail
