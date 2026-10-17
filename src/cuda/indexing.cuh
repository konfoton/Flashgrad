#pragma once

#include <array>
#include <cstdint>

#include "cuda/ops.h"
#include "flashgrad/core/shape.h"

namespace flashgrad::cuda::detail {

// A shape plus the strides of N operands over that shape, passed to kernels by value
// (no device allocation for metadata). Maps a row-major linear index to element offsets.
template <int N>
struct OffsetCalc {
  int ndim = 0;
  std::int64_t sizes[kMaxDims] = {};
  std::int64_t strides[N][kMaxDims] = {};

  // off[k] = element offset of linear index `i` in operand k.
  __device__ __forceinline__ void offsets(std::int64_t i, std::int64_t (&off)[N]) const {
#pragma unroll
    for (int k = 0; k < N; ++k) off[k] = 0;
    for (int d = ndim - 1; d >= 0; --d) {
      const std::int64_t pos = i % sizes[d];
      i /= sizes[d];
#pragma unroll
      for (int k = 0; k < N; ++k) off[k] += pos * strides[k][d];
    }
  }
};

// Offsets when every operand is contiguous.
template <int N>
struct IdentityCalc {
  __device__ __forceinline__ void offsets(std::int64_t i, std::int64_t (&off)[N]) const {
#pragma unroll
    for (int k = 0; k < N; ++k) off[k] = i;
  }
};

// Each strides[k] has shape.size() entries.
template <int N>
OffsetCalc<N> make_offset_calc(const Shape& shape, const std::array<const Strides*, N>& strides) {
  OffsetCalc<N> calc;
  calc.ndim = static_cast<int>(shape.size());
  for (int d = 0; d < calc.ndim; ++d) {
    calc.sizes[d] = shape[d];
    for (int k = 0; k < N; ++k) calc.strides[k][d] = (*strides[k])[d];
  }
  return calc;
}

}  // namespace flashgrad::cuda::detail
