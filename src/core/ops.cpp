#include "flashgrad/core/ops.h"

#include <cstdint>
#include <vector>

#include "autograd/function.h"
#include "core/checks.h"
#include "cuda/ops.h"
#include "flashgrad/autograd/grad_mode.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/shape.h"

namespace flashgrad {

namespace {

using autograd::detail::needs_grad;
using autograd::detail::record;
using cuda::detail::BinaryOp;
using cuda::detail::kMaxDims;
using cuda::detail::ReduceOp;
using cuda::detail::UnaryOp;
using Grads = std::vector<Tensor>;

void check_input(const char* op, const Tensor& t) {
  detail::check_defined(op, t);
  FG_CHECK(t.device().is_cuda(), op, ": expected a CUDA tensor (all math runs on the GPU), got ",
           describe(t));
  FG_CHECK(t.dim() <= kMaxDims, op, ": at most ", kMaxDims, " dimensions are supported, got ",
           describe(t));
}

Tensor unary(UnaryOp kind, const char* op, const Tensor& a) {
  check_input(op, a);
  Tensor out = empty(a.shape(), a.options());
  cuda::detail::unary(kind, out, a);
  return out;
}

Tensor binary(BinaryOp kind, const char* op, const Tensor& a, const Tensor& b) {
  check_input(op, a);
  check_input(op, b);
  FG_CHECK(a.device() == b.device(), op, ": operands are on different devices, got ", describe(a),
           " and ", describe(b));
  const Shape shape = broadcast_shapes(a.shape(), b.shape(), op);
  Tensor out = empty(shape, a.options());
  const NoGradGuard no_grad;  // the expanded operands are internal, not part of the graph
  cuda::detail::binary(kind, out, a.expand(shape), b.expand(shape));
  return out;
}

Tensor scalar_like(const char* op, const Tensor& like, float s) {
  detail::check_defined(op, like);
  return full({}, s, like.options());
}

Tensor reduce(ReduceOp kind, const char* op, const Tensor& a, std::int64_t dim, bool keepdim) {
  check_input(op, a);
  const std::int64_t d = normalize_dim(dim, a.dim(), op);
  FG_CHECK(kind != ReduceOp::Max || a.size(d) > 0, op, ": cannot reduce over empty dimension ",
           d, " of ", describe(a));
  Shape shape = a.shape();
  shape[d] = 1;
  Tensor out = empty(shape, a.options());
  cuda::detail::reduce(kind, out, a, d);
  return keepdim ? out : out.squeeze(d);
}

}  // namespace

// Each differentiable op computes its output, then records how to get the inputs' gradients
// from the output's gradient g (see autograd/function.h). Gradients may keep the broadcast
// shape of the output; the engine sums them back to each input's shape.

Tensor neg(const Tensor& a) {
  Tensor out = unary(UnaryOp::Neg, "neg", a);
  if (needs_grad({a})) record(out, "neg", {a}, [](const Tensor& g) { return Grads{-g}; });
  return out;
}

// The gradient passes where a > 0 and is 0 elsewhere, including at a == 0.
Tensor relu(const Tensor& a) {
  Tensor out = unary(UnaryOp::Relu, "relu", a);
  if (needs_grad({a})) {
    record(out, "relu", {a}, [a = a.detach()](const Tensor& g) {
      return Grads{g * lt(zeros({}, a.options()), a)};
    });
  }
  return out;
}

Tensor exp(const Tensor& a) {
  Tensor out = unary(UnaryOp::Exp, "exp", a);
  if (needs_grad({a})) {
    record(out, "exp", {a}, [out = out.detach()](const Tensor& g) { return Grads{g * out}; });
  }
  return out;
}

Tensor log(const Tensor& a) {
  Tensor out = unary(UnaryOp::Log, "log", a);
  if (needs_grad({a})) {
    record(out, "log", {a}, [a = a.detach()](const Tensor& g) { return Grads{g / a}; });
  }
  return out;
}

// d sigmoid(a) = g * s * (1 - s) with s = sigmoid(a), the saved output.
Tensor sigmoid(const Tensor& a) {
  Tensor out = unary(UnaryOp::Sigmoid, "sigmoid", a);
  if (needs_grad({a})) {
    record(out, "sigmoid", {a}, [s = out.detach()](const Tensor& g) {
      return Grads{g * s * (1.0f - s)};
    });
  }
  return out;
}

// d (1 / a) = -g / a^2 = -g * out^2.
Tensor reciprocal(const Tensor& a) {
  Tensor out = unary(UnaryOp::Reciprocal, "reciprocal", a);
  if (needs_grad({a})) {
    record(out, "reciprocal", {a}, [out = out.detach()](const Tensor& g) {
      return Grads{-(g * out * out)};
    });
  }
  return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out = binary(BinaryOp::Add, "add", a, b);
  if (needs_grad({a, b})) {
    record(out, "add", {a, b}, [](const Tensor& g) { return Grads{g, g}; });
  }
  return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
  Tensor out = binary(BinaryOp::Sub, "sub", a, b);
  if (needs_grad({a, b})) {
    record(out, "sub", {a, b}, [](const Tensor& g) { return Grads{g, -g}; });
  }
  return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  Tensor out = binary(BinaryOp::Mul, "mul", a, b);
  if (needs_grad({a, b})) {
    record(out, "mul", {a, b},
           [a = a.detach(), b = b.detach(), need_a = a.requires_grad(),
            need_b = b.requires_grad()](const Tensor& g) {
             return Grads{need_a ? g * b : Tensor(), need_b ? g * a : Tensor()};
           });
  }
  return out;
}
// For out = a / b: g / b for a, and -g * a / b^2 = -g * out / b for b.
Tensor div(const Tensor& a, const Tensor& b) {
  Tensor out = binary(BinaryOp::Div, "div", a, b);
  if (needs_grad({a, b})) {
    record(out, "div", {a, b},
           [b = b.detach(), out = out.detach(), need_a = a.requires_grad(),
            need_b = b.requires_grad()](const Tensor& g) {
             return Grads{need_a ? g / b : Tensor(), need_b ? -(g * out) / b : Tensor()};
           });
  }
  return out;
}
Tensor maximum(const Tensor& a, const Tensor& b) {
  return binary(BinaryOp::Maximum, "maximum", a, b);
}
Tensor lt(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Lt, "lt", a, b); }
Tensor eq(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Eq, "eq", a, b); }

Tensor sum(const Tensor& a, std::int64_t dim, bool keepdim) {
  return reduce(ReduceOp::Sum, "sum", a, dim, keepdim);
}
Tensor prod(const Tensor& a, std::int64_t dim, bool keepdim) {
  return reduce(ReduceOp::Prod, "prod", a, dim, keepdim);
}
Tensor max(const Tensor& a, std::int64_t dim, bool keepdim) {
  return reduce(ReduceOp::Max, "max", a, dim, keepdim);
}

Tensor sum(const Tensor& a) {
  check_input("sum", a);
  Tensor out = empty({}, a.options());
  cuda::detail::reduce_all(ReduceOp::Sum, out, a);
  return out;
}

Tensor mean(const Tensor& a) { return sum(a) / static_cast<float>(a.numel()); }

Tensor mean(const Tensor& a, std::int64_t dim, bool keepdim) {
  check_input("mean", a);
  const std::int64_t n = a.size(normalize_dim(dim, a.dim(), "mean"));
  return sum(a, dim, keepdim) / static_cast<float>(n);
}

Tensor sum_to(const Tensor& a, const Shape& shape) {
  check_input("sum_to", a);
  FG_CHECK(broadcast_shapes(shape, a.shape(), "sum_to") == a.shape(), "sum_to: ", describe(a),
           " is not a broadcast of shape ", to_string(shape));
  Tensor out = a;
  const auto lead = a.dim() - static_cast<std::int64_t>(shape.size());
  for (std::int64_t d = 0; d < lead; ++d) out = sum(out, 0);
  for (std::size_t d = 0; d < shape.size(); ++d) {
    if (shape[d] == 1 && out.size(static_cast<std::int64_t>(d)) != 1) {
      out = sum(out, static_cast<std::int64_t>(d), /*keepdim=*/true);
    }
  }
  return out;
}

Tensor log_softmax(const Tensor& a, std::int64_t dim) {
  check_input("log_softmax", a);
  const std::int64_t d = normalize_dim(dim, a.dim(), "log_softmax");
  Tensor out = empty(a.shape(), a.options());
  cuda::detail::log_softmax(out, a, d);
  return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  check_input("matmul", a);
  check_input("matmul", b);
  FG_CHECK(a.device() == b.device(), "matmul: operands are on different devices, got ",
           describe(a), " and ", describe(b));
  FG_CHECK(a.dim() == 2 && b.dim() == 2, "matmul: expected 2-D tensors, got ", describe(a),
           " and ", describe(b));
  FG_CHECK(a.size(1) == b.size(0), "matmul: inner dimensions don't match, got ", describe(a),
           " and ", describe(b));
  Tensor out = empty({a.size(0), b.size(1)}, a.options());
  cuda::detail::matmul(out, a, b);
  return out;
}

Tensor operator-(const Tensor& a) { return neg(a); }
Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }
Tensor operator+(const Tensor& a, float s) { return add(a, scalar_like("add", a, s)); }
Tensor operator-(const Tensor& a, float s) { return sub(a, scalar_like("sub", a, s)); }
Tensor operator*(const Tensor& a, float s) { return mul(a, scalar_like("mul", a, s)); }
Tensor operator/(const Tensor& a, float s) { return div(a, scalar_like("div", a, s)); }
Tensor operator+(float s, const Tensor& a) { return add(scalar_like("add", a, s), a); }
Tensor operator-(float s, const Tensor& a) { return sub(scalar_like("sub", a, s), a); }
Tensor operator*(float s, const Tensor& a) { return mul(scalar_like("mul", a, s), a); }
Tensor operator/(float s, const Tensor& a) { return div(scalar_like("div", a, s), a); }

}  // namespace flashgrad
