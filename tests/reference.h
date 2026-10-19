#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "flashgrad/core/random.h"
#include "flashgrad/core/shape.h"
#include "flashgrad/core/tensor.h"

// Naive host references for the ops tests: plain loops over row-major std::vectors with
// double arithmetic. Test code only, never part of the library.
namespace ref {

using flashgrad::Shape;

inline std::int64_t count(const Shape& shape) {
  std::int64_t n = 1;
  for (const std::int64_t s : shape) n *= s;
  return n;
}

inline std::vector<std::int64_t> row_major_strides(const Shape& shape) {
  std::vector<std::int64_t> strides(shape.size(), 1);
  for (std::size_t d = shape.size(); d-- > 1;) strides[d - 1] = strides[d] * shape[d];
  return strides;
}

template <typename F>
std::vector<float> unary(const std::vector<float>& a, F f) {
  std::vector<float> out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = static_cast<float>(f(double{a[i]}));
  return out;
}

// `a` and `b` hold row-major values of shapes `as` and `bs`; the result has the broadcast
// shape `os`.
template <typename F>
std::vector<float> binary(const std::vector<float>& a, const Shape& as, const std::vector<float>& b,
                          const Shape& bs, const Shape& os, F f) {
  const auto a_strides = row_major_strides(as);
  const auto b_strides = row_major_strides(bs);
  const auto ndim = static_cast<std::int64_t>(os.size());
  const auto a_lead = ndim - static_cast<std::int64_t>(as.size());
  const auto b_lead = ndim - static_cast<std::int64_t>(bs.size());
  std::vector<float> out(static_cast<std::size_t>(count(os)));
  for (std::int64_t i = 0; i < count(os); ++i) {
    std::int64_t rest = i, ai = 0, bi = 0;
    for (std::int64_t d = ndim - 1; d >= 0; --d) {
      const std::int64_t pos = rest % os[d];
      rest /= os[d];
      if (d >= a_lead && as[d - a_lead] != 1) ai += pos * a_strides[d - a_lead];
      if (d >= b_lead && bs[d - b_lead] != 1) bi += pos * b_strides[d - b_lead];
    }
    out[i] = static_cast<float>(f(double{a[ai]}, double{b[bi]}));
  }
  return out;
}

// Reduces row-major `a` of `shape` along `dim`; the result is row-major with keepdim layout.
template <typename F>
std::vector<float> reduce(const std::vector<float>& a, const Shape& shape, std::int64_t dim, F f,
                          double init) {
  std::int64_t outer = 1, inner = 1;
  for (std::int64_t d = 0; d < dim; ++d) outer *= shape[d];
  for (std::int64_t d = dim + 1; d < static_cast<std::int64_t>(shape.size()); ++d) {
    inner *= shape[d];
  }
  const std::int64_t n = shape[dim];
  std::vector<float> out(static_cast<std::size_t>(outer * inner));
  for (std::int64_t o = 0; o < outer; ++o) {
    for (std::int64_t in = 0; in < inner; ++in) {
      double acc = init;
      for (std::int64_t k = 0; k < n; ++k) acc = f(acc, double{a[(o * n + k) * inner + in]});
      out[o * inner + in] = static_cast<float>(acc);
    }
  }
  return out;
}

// Row-major `a` [m, k] times row-major `b` [k, n].
inline std::vector<float> matmul(const std::vector<float>& a, const std::vector<float>& b,
                                 std::int64_t m, std::int64_t k, std::int64_t n) {
  std::vector<float> out(static_cast<std::size_t>(m * n));
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (std::int64_t p = 0; p < k; ++p) acc += double{a[i * k + p]} * double{b[p * n + j]};
      out[i * n + j] = static_cast<float>(acc);
    }
  }
  return out;
}

inline double nan_max(double a, double b) { return (a > b || std::isnan(a)) ? a : b; }

// ---- Helpers --------------------------------------------------------------------------

// A random CUDA tensor; each standard normal value is passed through `transform` first.
template <typename T>
flashgrad::Tensor random_cuda(const Shape& shape, std::uint64_t seed, T transform) {
  flashgrad::Generator gen(seed);
  std::vector<float> values = flashgrad::randn(shape, {}, &gen).to_vector();
  for (float& v : values) v = transform(v);
  return flashgrad::tensor(values, shape).cuda();
}

inline flashgrad::Tensor random_cuda(const Shape& shape, std::uint64_t seed) {
  return random_cuda(shape, seed, [](float v) { return v; });
}

// Elementwise |actual - expected| <= atol + rtol * |expected|; NaN matches only NaN.
// rtol = atol = 0 asks for an exact match.
inline ::testing::AssertionResult close(const std::vector<float>& actual,
                                        const std::vector<float>& expected, double rtol = 0.0,
                                        double atol = 0.0) {
  if (actual.size() != expected.size()) {
    return ::testing::AssertionFailure()
           << "size " << actual.size() << " != expected size " << expected.size();
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double x = actual[i], y = expected[i];
    const bool ok = std::isnan(y) ? std::isnan(x)
                                  : (x == y || std::abs(x - y) <= atol + rtol * std::abs(y));
    if (!ok) {
      return ::testing::AssertionFailure()
             << "element " << i << ": got " << x << ", expected " << y << " (rtol=" << rtol
             << ", atol=" << atol << ")";
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace ref
