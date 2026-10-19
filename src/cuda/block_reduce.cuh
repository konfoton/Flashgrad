#pragma once

#include <algorithm>
#include <cstdint>

// Building blocks for kernels where one block combines one row of values: the reduction
// functors, a block-wide reduction and the block size for a given row length.
namespace flashgrad::cuda::detail {

inline constexpr int kWarpSize = 32;
inline constexpr int kMaxRowThreads = 256;  // multiple of 32, at most 32 warps

struct SumFn {
  static constexpr float kIdentity = 0.0f;
  __device__ float operator()(float a, float b) const { return a + b; }
};
struct ProdFn {
  static constexpr float kIdentity = 1.0f;
  __device__ float operator()(float a, float b) const { return a * b; }
};
struct MaxFn {
  static constexpr float kIdentity = -__builtin_huge_valf();
  __device__ float operator()(float a, float b) const { return (a > b || a != a) ? a : b; }
};

// Threads for a row of `row_size` elements: whole warps, enough to cover the row, 1 to 8.
inline std::uint32_t row_block_threads(std::int64_t row_size) {
  const auto warps = std::clamp<std::int64_t>((row_size + kWarpSize - 1) / kWarpSize, 1,
                                              kMaxRowThreads / kWarpSize);
  return static_cast<std::uint32_t>(warps * kWarpSize);
}

// Combines each thread's `v` over the block: a warp-shuffle tree, then warp 0 combines the
// per-warp partials. Every thread gets the result. blockDim.x must be a multiple of 32 and
// `scratch` must hold blockDim.x / 32 floats; all threads of the block must call it.
template <typename F>
__device__ float block_reduce(F f, float v, float* scratch) {
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int n_warps = static_cast<int>(blockDim.x) / kWarpSize;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v = f(v, __shfl_down_sync(0xffffffffu, v, offset));
  }
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < n_warps ? scratch[lane] : F::kIdentity;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      v = f(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  v = scratch[0];
  __syncthreads();  // scratch may be reused right after the call
  return v;
}

}  // namespace flashgrad::cuda::detail
