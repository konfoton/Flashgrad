#include "cpu/ops.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <type_traits>

#include "cpu/loops.h"

namespace flashgrad::cpu {

namespace {

using detail::for_each_offset;
using Strides1 = std::array<std::span<const std::int64_t>, 1>;
using Strides2 = std::array<std::span<const std::int64_t>, 2>;
using Strides3 = std::array<std::span<const std::int64_t>, 3>;

// Accumulator type for sums: double for floating point, int64 for integers.
template <typename T>
using Acc = std::conditional_t<std::is_floating_point_v<T>, double, std::int64_t>;

template <typename T>
bool is_nan(T x) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::isnan(x);
  } else {
    return false;
  }
}

// out[i] = f(a[i]) over same-shape tensors.
template <typename Out, typename In, typename F>
void map1(const Tensor& out, const Tensor& a, F f) {
  Out* o = out.data_ptr<Out>();
  const In* x = a.data_ptr<In>();
  for_each_offset<2>(out.shape(), Strides2{out.strides(), a.strides()},
                     [&](const auto& off) { o[off[0]] = f(x[off[1]]); });
}

// out[i] = f(a[i], b[i]) over same-shape tensors.
template <typename Out, typename In, typename F>
void map2(const Tensor& out, const Tensor& a, const Tensor& b, F f) {
  Out* o = out.data_ptr<Out>();
  const In* x = a.data_ptr<In>();
  const In* y = b.data_ptr<In>();
  for_each_offset<3>(out.shape(), Strides3{out.strides(), a.strides(), b.strides()},
                     [&](const auto& off) { o[off[0]] = f(x[off[1]], y[off[2]]); });
}

// Calls f(out_offset, in_offset, inner_shape, inner_strides) for every element of `out`,
// where `in` has out's dims followed by the dims to reduce over.
template <typename F>
void for_each_reduction(const Tensor& out, const Tensor& in, F f) {
  const auto outer = static_cast<std::size_t>(out.dim());
  const std::span<const std::int64_t> in_shape(in.shape());
  const std::span<const std::int64_t> in_strides(in.strides());
  for_each_offset<2>(out.shape(), Strides2{out.strides(), in_strides.first(outer)},
                     [&](const auto& off) {
                       f(off[0], off[1], in_shape.subspan(outer), in_strides.subspan(outer));
                     });
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

void binary(BinaryOp op, const Tensor& out, const Tensor& a, const Tensor& b) {
  switch (op) {
    case BinaryOp::Add:
      return dispatch_numeric(out.dtype(), "add", [&]<typename T>() {
        map2<T, T>(out, a, b, [](T x, T y) -> T { return x + y; });
      });
    case BinaryOp::Sub:
      return dispatch_numeric(out.dtype(), "sub", [&]<typename T>() {
        map2<T, T>(out, a, b, [](T x, T y) -> T { return x - y; });
      });
    case BinaryOp::Mul:
      return dispatch_numeric(out.dtype(), "mul", [&]<typename T>() {
        map2<T, T>(out, a, b, [](T x, T y) -> T { return x * y; });
      });
    case BinaryOp::Div:
      return dispatch_numeric(out.dtype(), "div", [&]<typename T>() {
        map2<T, T>(out, a, b, [](T x, T y) -> T {
          if constexpr (std::is_integral_v<T>) {
            FG_CHECK(y != 0, "div: integer division by zero");
          }
          return x / y;
        });
      });
    case BinaryOp::Maximum:
      return dispatch_numeric(out.dtype(), "maximum", [&]<typename T>() {
        map2<T, T>(out, a, b, [](T x, T y) -> T {
          if (is_nan(x)) return x;
          if (is_nan(y)) return y;
          return x > y ? x : y;
        });
      });
    case BinaryOp::Minimum:
      return dispatch_numeric(out.dtype(), "minimum", [&]<typename T>() {
        map2<T, T>(out, a, b, [](T x, T y) -> T {
          if (is_nan(x)) return x;
          if (is_nan(y)) return y;
          return x < y ? x : y;
        });
      });
    case BinaryOp::Pow:
      return dispatch_floating(out.dtype(), "pow", [&]<typename T>() {
        map2<T, T>(out, a, b, [](T x, T y) -> T { return std::pow(x, y); });
      });
  }
}

void compare(CompareOp op, const Tensor& out, const Tensor& a, const Tensor& b) {
  dispatch_all(a.dtype(), "compare", [&]<typename T>() {
    switch (op) {
      case CompareOp::Eq:
        return map2<bool, T>(out, a, b, [](T x, T y) { return x == y; });
      case CompareOp::Ne:
        return map2<bool, T>(out, a, b, [](T x, T y) { return x != y; });
      case CompareOp::Lt:
        return map2<bool, T>(out, a, b, [](T x, T y) { return x < y; });
      case CompareOp::Le:
        return map2<bool, T>(out, a, b, [](T x, T y) { return x <= y; });
      case CompareOp::Gt:
        return map2<bool, T>(out, a, b, [](T x, T y) { return x > y; });
      case CompareOp::Ge:
        return map2<bool, T>(out, a, b, [](T x, T y) { return x >= y; });
    }
  });
}

void unary(UnaryOp op, const Tensor& out, const Tensor& a) {
  switch (op) {
    case UnaryOp::Neg:
      return dispatch_numeric(out.dtype(), "neg", [&]<typename T>() {
        map1<T, T>(out, a, [](T x) -> T { return -x; });
      });
    case UnaryOp::Abs:
      return dispatch_numeric(out.dtype(), "abs", [&]<typename T>() {
        map1<T, T>(out, a, [](T x) -> T { return std::abs(x); });
      });
    case UnaryOp::Relu:
      return dispatch_numeric(out.dtype(), "relu", [&]<typename T>() {
        // Written so that NaN propagates.
        map1<T, T>(out, a, [](T x) -> T { return x < T(0) ? T(0) : x; });
      });
    case UnaryOp::Exp:
      return dispatch_floating(out.dtype(), "exp", [&]<typename T>() {
        map1<T, T>(out, a, [](T x) -> T { return std::exp(x); });
      });
    case UnaryOp::Log:
      return dispatch_floating(out.dtype(), "log", [&]<typename T>() {
        map1<T, T>(out, a, [](T x) -> T { return std::log(x); });
      });
    case UnaryOp::Sqrt:
      return dispatch_floating(out.dtype(), "sqrt", [&]<typename T>() {
        map1<T, T>(out, a, [](T x) -> T { return std::sqrt(x); });
      });
    case UnaryOp::Tanh:
      return dispatch_floating(out.dtype(), "tanh", [&]<typename T>() {
        map1<T, T>(out, a, [](T x) -> T { return std::tanh(x); });
      });
    case UnaryOp::Sigmoid:
      return dispatch_floating(out.dtype(), "sigmoid", [&]<typename T>() {
        map1<T, T>(out, a, [](T x) -> T { return T(1) / (T(1) + std::exp(-x)); });
      });
  }
}

void reduce(ReduceOp op, const Tensor& out, const Tensor& in) {
  dispatch_numeric(in.dtype(), "reduce", [&]<typename T>() {
    T* o = out.data_ptr<T>();
    const T* x = in.data_ptr<T>();
    for_each_reduction(
        out, in, [&](std::int64_t out_off, std::int64_t in_off, auto shape, auto strides) {
          switch (op) {
            case ReduceOp::Sum:
            case ReduceOp::Mean: {
              Acc<T> acc = 0;
              std::int64_t count = 0;
              for_each_offset<1>(shape, Strides1{strides}, [&](const auto& off) {
                acc += static_cast<Acc<T>>(x[in_off + off[0]]);
                ++count;
              });
              if (op == ReduceOp::Sum) {
                o[out_off] = static_cast<T>(acc);
              } else {
                // Mean of zero elements is 0 / 0 = NaN, as in PyTorch.
                o[out_off] = static_cast<T>(static_cast<double>(acc) / static_cast<double>(count));
              }
              return;
            }
            case ReduceOp::Max:
            case ReduceOp::Min: {
              const bool is_max = op == ReduceOp::Max;
              bool first = true;
              T best{};
              for_each_offset<1>(shape, Strides1{strides}, [&](const auto& off) {
                const T v = x[in_off + off[0]];
                if (first || is_nan(v)) {
                  if (first || !is_nan(best)) best = v;
                  first = false;
                } else if (!is_nan(best) && (is_max ? v > best : v < best)) {
                  best = v;
                }
              });
              FG_CHECK(!first, "reduce: max/min over zero elements");
              o[out_off] = best;
              return;
            }
          }
        });
  });
}

void arg_reduce(ArgReduceOp op, const Tensor& out, const Tensor& in) {
  dispatch_numeric(in.dtype(), "arg_reduce", [&]<typename T>() {
    std::int64_t* o = out.data_ptr<std::int64_t>();
    const T* x = in.data_ptr<T>();
    const bool is_max = op == ArgReduceOp::Argmax;
    for_each_reduction(out, in,
                       [&](std::int64_t out_off, std::int64_t in_off, auto shape, auto strides) {
                         std::int64_t index = 0;
                         std::int64_t best_index = -1;
                         T best{};
                         // The first NaN wins (NaN is treated as larger/smaller than everything, as
                         // in PyTorch).
                         for_each_offset<1>(shape, Strides1{strides}, [&](const auto& off) {
                           const T v = x[in_off + off[0]];
                           if (best_index < 0 ||
                               (!is_nan(best) && (is_nan(v) || (is_max ? v > best : v < best)))) {
                             best = v;
                             best_index = index;
                           }
                           ++index;
                         });
                         FG_CHECK(best_index >= 0, "arg_reduce: argmax/argmin over zero elements");
                         o[out_off] = best_index;
                       });
  });
}

void matmul(const Tensor& out, const Tensor& a, const Tensor& b) {
  const auto nb = static_cast<std::size_t>(out.dim()) - 2;  // number of batch dims
  const std::int64_t m = out.shape()[nb];
  const std::int64_t n = out.shape()[nb + 1];
  const std::int64_t k = a.shape()[nb + 1];
  const std::int64_t a_row = a.strides()[nb], a_col = a.strides()[nb + 1];
  const std::int64_t b_row = b.strides()[nb], b_col = b.strides()[nb + 1];
  const std::int64_t o_row = out.strides()[nb], o_col = out.strides()[nb + 1];
  dispatch_numeric(out.dtype(), "matmul", [&]<typename T>() {
    T* o = out.data_ptr<T>();
    const T* x = a.data_ptr<T>();
    const T* y = b.data_ptr<T>();
    const std::span<const std::int64_t> batch(out.shape().data(), nb);
    for_each_offset<3>(batch,
                       Strides3{std::span(out.strides()).first(nb),
                                std::span(a.strides()).first(nb), std::span(b.strides()).first(nb)},
                       [&](const auto& off) {
                         for (std::int64_t i = 0; i < m; ++i) {
                           for (std::int64_t j = 0; j < n; ++j) {
                             Acc<T> acc = 0;
                             for (std::int64_t p = 0; p < k; ++p) {
                               acc += static_cast<Acc<T>>(x[off[1] + i * a_row + p * a_col]) *
                                      static_cast<Acc<T>>(y[off[2] + p * b_row + j * b_col]);
                             }
                             o[off[0] + i * o_row + j * o_col] = static_cast<T>(acc);
                           }
                         }
                       });
  });
}

void softmax(const Tensor& out, const Tensor& in, bool log_space) {
  const auto outer = static_cast<std::size_t>(in.dim()) - 1;
  const std::int64_t len = in.shape().back();
  const std::int64_t in_stride = in.strides().back();
  const std::int64_t out_stride = out.strides().back();
  if (in.numel() == 0) return;  // also avoids offsetting the null data pointer
  dispatch_floating(in.dtype(), "softmax", [&]<typename T>() {
    T* o = out.data_ptr<T>();
    const T* x = in.data_ptr<T>();
    for_each_offset<2>(
        std::span(in.shape()).first(outer),
        Strides2{std::span(out.strides()).first(outer), std::span(in.strides()).first(outer)},
        [&](const auto& off) {
          const T* row = x + off[1];
          T* dst = o + off[0];
          // NaN inputs make the whole row NaN through the sum below.
          double row_max = -std::numeric_limits<double>::infinity();
          for (std::int64_t j = 0; j < len; ++j) {
            row_max = std::fmax(row_max, static_cast<double>(row[j * in_stride]));
          }
          double row_sum = 0.0;
          for (std::int64_t j = 0; j < len; ++j) {
            row_sum += std::exp(static_cast<double>(row[j * in_stride]) - row_max);
          }
          const double log_sum = std::log(row_sum);
          for (std::int64_t j = 0; j < len; ++j) {
            const double shifted = static_cast<double>(row[j * in_stride]) - row_max;
            dst[j * out_stride] =
                static_cast<T>(log_space ? shifted - log_sum : std::exp(shifted) / row_sum);
          }
        });
  });
}

}  // namespace flashgrad::cpu
