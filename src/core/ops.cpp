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
Tensor mul(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Mul, "mul", a, b); }
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

}  // namespace flashgrad
