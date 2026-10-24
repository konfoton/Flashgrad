#include "flashgrad/core/ops.h"

#include <cstdint>

#include "core/checks.h"
#include "cuda/ops.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/shape.h"

namespace flashgrad {

namespace {

using cuda::detail::BinaryOp;
using cuda::detail::kMaxDims;
using cuda::detail::ReduceOp;
using cuda::detail::UnaryOp;

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

Tensor neg(const Tensor& a) { return unary(UnaryOp::Neg, "neg", a); }
Tensor relu(const Tensor& a) { return unary(UnaryOp::Relu, "relu", a); }
Tensor exp(const Tensor& a) { return unary(UnaryOp::Exp, "exp", a); }
Tensor log(const Tensor& a) { return unary(UnaryOp::Log, "log", a); }
Tensor sigmoid(const Tensor& a) { return unary(UnaryOp::Sigmoid, "sigmoid", a); }
Tensor reciprocal(const Tensor& a) { return unary(UnaryOp::Reciprocal, "reciprocal", a); }

Tensor add(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Add, "add", a, b); }
Tensor sub(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Sub, "sub", a, b); }
Tensor mul(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Mul, "mul", a, b); }
Tensor div(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Div, "div", a, b); }
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
  FG_CHECK(a.dim() == 2 && b.dim() == 2, "matmul: expected 2-D matrices, got ", describe(a),
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
