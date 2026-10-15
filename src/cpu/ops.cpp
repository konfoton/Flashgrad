#include "cpu/ops.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>

#include "cpu/loops.h"

namespace flashgrad::cpu {

namespace {

using detail::for_each_offset;
using Strides1 = std::array<std::span<const std::int64_t>, 1>;
using Strides2 = std::array<std::span<const std::int64_t>, 2>;
using Strides3 = std::array<std::span<const std::int64_t>, 3>;

// out[i] = f(a[i]) over same-shape tensors.
template <typename F>
void map1(const Tensor& out, const Tensor& a, F f) {
  float* o = out.data_ptr();
  const float* x = a.data_ptr();
  for_each_offset<2>(out.shape(), Strides2{out.strides(), a.strides()},
                     [&](const auto& off) { o[off[0]] = f(x[off[1]]); });
}

// out[i] = f(a[i], b[i]) over same-shape tensors.
template <typename F>
void map2(const Tensor& out, const Tensor& a, const Tensor& b, F f) {
  float* o = out.data_ptr();
  const float* x = a.data_ptr();
  const float* y = b.data_ptr();
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

float from_bool(bool b) { return b ? 1.0f : 0.0f; }

}  // namespace

void copy(const Tensor& dst, const Tensor& src) {
  map1(dst, src, [](float x) { return x; });
}

void fill(const Tensor& out, double value) {
  float* o = out.data_ptr();
  const auto v = static_cast<float>(value);
  for_each_offset<1>(out.shape(), Strides1{out.strides()}, [&](const auto& off) { o[off[0]] = v; });
}

void binary(BinaryOp op, const Tensor& out, const Tensor& a, const Tensor& b) {
  switch (op) {
    case BinaryOp::Add:
      return map2(out, a, b, [](float x, float y) { return x + y; });
    case BinaryOp::Sub:
      return map2(out, a, b, [](float x, float y) { return x - y; });
    case BinaryOp::Mul:
      return map2(out, a, b, [](float x, float y) { return x * y; });
    case BinaryOp::Div:
      return map2(out, a, b, [](float x, float y) { return x / y; });
    case BinaryOp::Maximum:
      return map2(out, a, b, [](float x, float y) {
        if (std::isnan(x)) return x;
        if (std::isnan(y)) return y;
        return x > y ? x : y;
      });
    case BinaryOp::Minimum:
      return map2(out, a, b, [](float x, float y) {
        if (std::isnan(x)) return x;
        if (std::isnan(y)) return y;
        return x < y ? x : y;
      });
    case BinaryOp::Pow:
      return map2(out, a, b, [](float x, float y) { return std::pow(x, y); });
  }
}

void compare(CompareOp op, const Tensor& out, const Tensor& a, const Tensor& b) {
  switch (op) {
    case CompareOp::Eq:
      return map2(out, a, b, [](float x, float y) { return from_bool(x == y); });
    case CompareOp::Ne:
      return map2(out, a, b, [](float x, float y) { return from_bool(x != y); });
    case CompareOp::Lt:
      return map2(out, a, b, [](float x, float y) { return from_bool(x < y); });
    case CompareOp::Le:
      return map2(out, a, b, [](float x, float y) { return from_bool(x <= y); });
    case CompareOp::Gt:
      return map2(out, a, b, [](float x, float y) { return from_bool(x > y); });
    case CompareOp::Ge:
      return map2(out, a, b, [](float x, float y) { return from_bool(x >= y); });
  }
}

void unary(UnaryOp op, const Tensor& out, const Tensor& a) {
  switch (op) {
    case UnaryOp::Neg:
      return map1(out, a, [](float x) { return -x; });
    case UnaryOp::Abs:
      return map1(out, a, [](float x) { return std::abs(x); });
    case UnaryOp::Relu:
      // Written so that NaN propagates.
      return map1(out, a, [](float x) { return x < 0.0f ? 0.0f : x; });
    case UnaryOp::Exp:
      return map1(out, a, [](float x) { return std::exp(x); });
    case UnaryOp::Log:
      return map1(out, a, [](float x) { return std::log(x); });
    case UnaryOp::Sqrt:
      return map1(out, a, [](float x) { return std::sqrt(x); });
    case UnaryOp::Tanh:
      return map1(out, a, [](float x) { return std::tanh(x); });
    case UnaryOp::Sigmoid:
      return map1(out, a, [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
  }
}

void reduce(ReduceOp op, const Tensor& out, const Tensor& in) {
  float* o = out.data_ptr();
  const float* x = in.data_ptr();
  for_each_reduction(
      out, in, [&](std::int64_t out_off, std::int64_t in_off, auto shape, auto strides) {
        switch (op) {
          case ReduceOp::Sum:
          case ReduceOp::Mean: {
            double acc = 0.0;
            std::int64_t count = 0;
            for_each_offset<1>(shape, Strides1{strides}, [&](const auto& off) {
              acc += static_cast<double>(x[in_off + off[0]]);
              ++count;
            });
            // Mean of zero elements is 0 / 0 = NaN, as in PyTorch.
            o[out_off] = static_cast<float>(
                op == ReduceOp::Sum ? acc : acc / static_cast<double>(count));
            return;
          }
          case ReduceOp::Max:
          case ReduceOp::Min: {
            const bool is_max = op == ReduceOp::Max;
            bool first = true;
            float best = 0.0f;
            for_each_offset<1>(shape, Strides1{strides}, [&](const auto& off) {
              const float v = x[in_off + off[0]];
              if (first || std::isnan(v)) {
                if (first || !std::isnan(best)) best = v;
                first = false;
              } else if (!std::isnan(best) && (is_max ? v > best : v < best)) {
                best = v;
              }
            });
            FG_CHECK(!first, "reduce: max/min over zero elements");
            o[out_off] = best;
            return;
          }
        }
      });
}

void arg_reduce(ArgReduceOp op, const Tensor& out, const Tensor& in) {
  float* o = out.data_ptr();
  const float* x = in.data_ptr();
  const bool is_max = op == ArgReduceOp::Argmax;
  for_each_reduction(
      out, in, [&](std::int64_t out_off, std::int64_t in_off, auto shape, auto strides) {
        std::int64_t index = 0;
        std::int64_t best_index = -1;
        float best = 0.0f;
        // The first NaN wins (NaN is treated as larger/smaller than everything, as in PyTorch).
        for_each_offset<1>(shape, Strides1{strides}, [&](const auto& off) {
          const float v = x[in_off + off[0]];
          if (best_index < 0 ||
              (!std::isnan(best) && (std::isnan(v) || (is_max ? v > best : v < best)))) {
            best = v;
            best_index = index;
          }
          ++index;
        });
        FG_CHECK(best_index >= 0, "arg_reduce: argmax/argmin over zero elements");
        o[out_off] = static_cast<float>(best_index);
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
  float* o = out.data_ptr();
  const float* x = a.data_ptr();
  const float* y = b.data_ptr();
  const std::span<const std::int64_t> batch(out.shape().data(), nb);
  for_each_offset<3>(batch,
                     Strides3{std::span(out.strides()).first(nb), std::span(a.strides()).first(nb),
                              std::span(b.strides()).first(nb)},
                     [&](const auto& off) {
                       for (std::int64_t i = 0; i < m; ++i) {
                         for (std::int64_t j = 0; j < n; ++j) {
                           double acc = 0.0;
                           for (std::int64_t p = 0; p < k; ++p) {
                             acc += static_cast<double>(x[off[1] + i * a_row + p * a_col]) *
                                    static_cast<double>(y[off[2] + p * b_row + j * b_col]);
                           }
                           o[off[0] + i * o_row + j * o_col] = static_cast<float>(acc);
                         }
                       }
                     });
}

void softmax(const Tensor& out, const Tensor& in, bool log_space) {
  const auto outer = static_cast<std::size_t>(in.dim()) - 1;
  const std::int64_t len = in.shape().back();
  const std::int64_t in_stride = in.strides().back();
  const std::int64_t out_stride = out.strides().back();
  if (in.numel() == 0) return;  // also avoids offsetting the null data pointer
  float* o = out.data_ptr();
  const float* x = in.data_ptr();
  for_each_offset<2>(
      std::span(in.shape()).first(outer),
      Strides2{std::span(out.strides()).first(outer), std::span(in.strides()).first(outer)},
      [&](const auto& off) {
        const float* row = x + off[1];
        float* dst = o + off[0];
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
              static_cast<float>(log_space ? shifted - log_sum : std::exp(shifted) / row_sum);
        }
      });
}

}  // namespace flashgrad::cpu
