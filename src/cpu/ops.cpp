#include "cpu/ops.h"

#include <array>
#include <cstdint>
#include <span>

#include "cpu/loops.h"

namespace flashgrad::cpu {

namespace {

using detail::for_each_offset;
using Strides1 = std::array<std::span<const std::int64_t>, 1>;
using Strides2 = std::array<std::span<const std::int64_t>, 2>;

// out[i] = f(a[i]) over same-shape tensors.
template <typename F>
void map1(const Tensor& out, const Tensor& a, F f) {
  float* o = out.data_ptr();
  const float* x = a.data_ptr();
  for_each_offset<2>(out.shape(), Strides2{out.strides(), a.strides()},
                     [&](const auto& off) { o[off[0]] = f(x[off[1]]); });
}

}  // namespace

void copy(const Tensor& dst, const Tensor& src) {
  map1(dst, src, [](float x) { return x; });
}

void fill(const Tensor& out, double value) {
  float* o = out.data_ptr();
  const auto v = static_cast<float>(value);
  for_each_offset<1>(out.shape(), Strides1{out.strides()}, [&](const auto& off) { o[off[0]] = v; });
}

}  // namespace flashgrad::cpu
