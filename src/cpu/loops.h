#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace flashgrad::cpu::detail {

// Visits every multi-index of `shape` in row-major order and calls f(offsets), where
// offsets[k] = sum_d index[d] * strides[k][d] is the element offset of the current index in
// operand k. Each strides[k] needs at least shape.size() entries (extra trailing entries are
// ignored, so a prefix of a tensor's dimensions can be iterated). Does nothing when the shape
// has no elements; calls f once for a 0-dimensional shape.
template <std::size_t N, typename F>
void for_each_offset(std::span<const std::int64_t> shape,
                     const std::array<std::span<const std::int64_t>, N>& strides, F&& f) {
  std::int64_t total = 1;
  for (const std::int64_t size : shape) total *= size;
  if (total == 0) return;

  const std::size_t ndim = shape.size();
  std::vector<std::int64_t> index(ndim, 0);
  std::array<std::int64_t, N> offsets{};
  for (std::int64_t n = 0; n < total; ++n) {
    f(static_cast<const std::array<std::int64_t, N>&>(offsets));
    // Increment the multi-index like an odometer, updating the offsets incrementally.
    for (std::size_t d = ndim; d-- > 0;) {
      ++index[d];
      for (std::size_t k = 0; k < N; ++k) offsets[k] += strides[k][d];
      if (index[d] < shape[d]) break;
      for (std::size_t k = 0; k < N; ++k) offsets[k] -= strides[k][d] * shape[d];
      index[d] = 0;
    }
  }
}

}  // namespace flashgrad::cpu::detail
