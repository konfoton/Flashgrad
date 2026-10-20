#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"

// Finite-difference gradient checks. The analytic gradients come from backward() on the
// GPU in float32; the numeric ones from central differences of the host reference
// (reference.h) evaluated in double. Test code only.
namespace gradcheck {

using flashgrad::Tensor;
using Values = std::vector<double>;

// The op under test, applied to leaves that require grad.
using Op = std::function<Tensor(const std::vector<Tensor>& inputs)>;
// The same function on row-major host values; returns the output in row-major order.
using Reference = std::function<Values(const std::vector<Values>& inputs)>;

struct Options {
  double eps = 1e-6;  // step of the central differences, in double
  double rtol = 1e-4;
  double atol = 1e-5;
  std::vector<bool> requires_grad;  // per input; empty: every input
  std::uint64_t seed = 1234;        // for the random output gradient
};

inline double dot(const Values& a, const Values& b) {
  double total = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) total += a[i] * b[i];
  return total;
}

inline Values to_values(const Tensor& t) {
  const std::vector<float> v = t.to_vector();
  return Values(v.begin(), v.end());
}

// Backpropagates a random gradient v through op(inputs) and compares each input's gradient
// with d(v . reference(x)) / dx, which is what backward computes. Inputs may be any views.
inline ::testing::AssertionResult check(const Op& op, const Reference& reference,
                                        const std::vector<Tensor>& inputs,
                                        const Options& options = {}) {
  std::vector<Tensor> leaves;
  std::vector<Values> x;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const bool wants_grad = options.requires_grad.empty() || options.requires_grad[i];
    leaves.push_back(inputs[i].detach().requires_grad_(wants_grad));
    x.push_back(to_values(inputs[i]));
  }

  const Tensor out = op(leaves);
  const Values expected = reference(x);
  if (static_cast<std::int64_t>(expected.size()) != out.numel()) {
    return ::testing::AssertionFailure() << "the reference returned " << expected.size()
                                         << " values for an output " << describe(out);
  }
  const Values actual = to_values(out);
  for (std::size_t j = 0; j < actual.size(); ++j) {
    if (std::abs(actual[j] - expected[j]) > 1e-3 + 1e-3 * std::abs(expected[j])) {
      return ::testing::AssertionFailure() << "forward value " << j << " is " << actual[j]
                                           << " but the reference gives " << expected[j];
    }
  }

  flashgrad::Generator gen(options.seed);
  const Tensor v = flashgrad::randn(out.shape(), {}, &gen).to(out.device());
  out.backward(v);
  const Values vv = to_values(v);

  for (std::size_t i = 0; i < leaves.size(); ++i) {
    if (!leaves[i].requires_grad()) {
      if (leaves[i].grad().defined()) {
        return ::testing::AssertionFailure() << "input " << i << " got a gradient";
      }
      continue;
    }
    const Tensor grad = leaves[i].grad();
    if (grad.defined() && grad.shape() != leaves[i].shape()) {
      return ::testing::AssertionFailure() << "the gradient of input " << i << " is "
                                           << describe(grad) << " for " << describe(leaves[i]);
    }
    const Values analytic = grad.defined() ? to_values(grad) : Values(x[i].size(), 0.0);
    for (std::size_t j = 0; j < x[i].size(); ++j) {
      const double original = x[i][j];
      x[i][j] = original + options.eps;
      const double plus = dot(reference(x), vv);
      x[i][j] = original - options.eps;
      const double minus = dot(reference(x), vv);
      x[i][j] = original;
      const double numeric = (plus - minus) / (2.0 * options.eps);
      if (std::abs(analytic[j] - numeric) > options.atol + options.rtol * std::abs(numeric)) {
        return ::testing::AssertionFailure()
               << "input " << i << " " << describe(leaves[i]) << ", element " << j
               << ": backward gives " << analytic[j] << ", finite differences give " << numeric
               << " (rtol=" << options.rtol << ", atol=" << options.atol << ")";
      }
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace gradcheck
