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
template <typename Out, typename In, typename F>
void map1(const Tensor& out, const Tensor& a, F f) {
  Out* o = out.data_ptr<Out>();
  const In* x = a.data_ptr<In>();
  for_each_offset<2>(out.shape(), Strides2{out.strides(), a.strides()},
                     [&](const auto& off) { o[off[0]] = f(x[off[1]]); });
}

}  // namespace

void copy(const Tensor& dst, const Tensor& src) {
  dispatch_all(dst.dtype(), "copy", [&]<typename D>() {
    dispatch_all(src.dtype(), "copy", [&]<typename S>() {
      map1<D, S>(dst, src, [](S x) { return static_cast<D>(x); });
    });
  });
}

void fill(const Tensor& out, double value) {
  dispatch_all(out.dtype(), "fill", [&]<typename T>() {
    T* o = out.data_ptr<T>();
    const auto v = static_cast<T>(value);
    for_each_offset<1>(out.shape(), Strides1{out.strides()},
                       [&](const auto& off) { o[off[0]] = v; });
  });
}

}  // namespace flashgrad::cpu
