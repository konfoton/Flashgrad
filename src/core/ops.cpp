#include "flashgrad/core/ops.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "core/checks.h"
#include "core/op_kinds.h"
#include "cpu/ops.h"

// Device-independent part of every op: argument validation, output shape/dtype and
// allocation, broadcasting (via expand views) and moving the reduced dim last (via permute).
// The backends then only see same-shape, already-allocated tensors.
namespace flashgrad {

using detail::check_defined;
using detail::check_floating;
using detail::check_numeric;
using detail::check_same_device_and_dtype;
using detail::run_on_device;

namespace {

Tensor scalar_like(double value, const Tensor& t) {
  check_defined("scalar", t);
  return full({}, value, t.options());
}

void check_binary_dtype(BinaryOp op, const char* name, const Tensor& t) {
  if (op == BinaryOp::Pow) {
    check_floating(name, t);
  } else {
    check_numeric(name, t);
  }
}

Tensor binary(BinaryOp op, const char* name, const Tensor& a, const Tensor& b) {
  check_same_device_and_dtype(name, a, b);
  check_binary_dtype(op, name, a);
  const Shape shape = broadcast_shapes(a.shape(), b.shape(), name);
  Tensor out = empty(shape, a.options());
  run_on_device(name, out, [&] { cpu::binary(op, out, a.expand(shape), b.expand(shape)); });
  return out;
}

const Tensor& binary_inplace(BinaryOp op, const char* name, const Tensor& self,
                             const Tensor& other) {
  check_same_device_and_dtype(name, self, other);
  check_binary_dtype(op, name, self);
  const Shape shape = broadcast_shapes(self.shape(), other.shape(), name);
  FG_CHECK(shape == self.shape(), name, ": broadcast result ", to_string(shape),
           " does not fit the in-place destination ", describe(self));
  detail::check_writable(name, self);
  // Each element of `self` is read and then written at the same position, which is safe;
  // a different view of the same memory could be overwritten before it is read.
  const Tensor rhs =
      other.storage().is_same(self.storage()) && !other.is_same(self) ? other.clone() : other;
  run_on_device(name, self, [&] { cpu::binary(op, self, self, rhs.expand(shape)); });
  return self;
}

Tensor compare(CompareOp op, const char* name, const Tensor& a, const Tensor& b) {
  check_same_device_and_dtype(name, a, b);
  const Shape shape = broadcast_shapes(a.shape(), b.shape(), name);
  Tensor out = empty(shape, {.dtype = DType::Bool, .device = a.device()});
  run_on_device(name, out, [&] { cpu::compare(op, out, a.expand(shape), b.expand(shape)); });
  return out;
}

Tensor unary(UnaryOp op, const char* name, const Tensor& a) {
  check_defined(name, a);
  switch (op) {
    case UnaryOp::Neg:
    case UnaryOp::Abs:
    case UnaryOp::Relu:
      check_numeric(name, a);
      break;
    case UnaryOp::Exp:
    case UnaryOp::Log:
    case UnaryOp::Sqrt:
    case UnaryOp::Tanh:
    case UnaryOp::Sigmoid:
      check_floating(name, a);
      break;
  }
  Tensor out = empty(a.shape(), a.options());
  run_on_device(name, out, [&] { cpu::unary(op, out, a); });
  return out;
}

void check_reduce_dtype(ReduceOp op, const char* name, const Tensor& a) {
  if (op == ReduceOp::Mean) {
    check_floating(name, a);
  } else {
    check_numeric(name, a);
  }
}

// Permutation that moves `dim` to the end, keeping the order of the other dims.
std::vector<std::int64_t> move_to_end(std::int64_t dim, std::int64_t ndim) {
  std::vector<std::int64_t> perm;
  perm.reserve(static_cast<std::size_t>(ndim));
  for (std::int64_t d = 0; d < ndim; ++d) {
    if (d != dim) perm.push_back(d);
  }
  perm.push_back(dim);
  return perm;
}

Tensor reduce_all(ReduceOp op, const char* name, const Tensor& a) {
  check_defined(name, a);
  check_reduce_dtype(op, name, a);
  FG_CHECK(a.numel() > 0 || op == ReduceOp::Sum || op == ReduceOp::Mean, name,
           ": cannot reduce over zero elements, got ", describe(a));
  Tensor out = empty({}, a.options());
  run_on_device(name, out, [&] { cpu::reduce(op, out, a); });
  return out;
}

Tensor reduce_dim(ReduceOp op, const char* name, const Tensor& a, std::int64_t dim, bool keepdim) {
  check_defined(name, a);
  check_reduce_dtype(op, name, a);
  const std::int64_t d = normalize_dim(dim, a.dim(), name);
  Shape shape = a.shape();
  shape.erase(shape.begin() + d);
  FG_CHECK(a.size(d) > 0 || numel(shape) == 0 || op == ReduceOp::Sum || op == ReduceOp::Mean, name,
           ": cannot reduce over zero elements (dimension ", dim, " of ", describe(a), ")");
  Tensor out = empty(shape, a.options());
  run_on_device(name, out, [&] { cpu::reduce(op, out, a.permute(move_to_end(d, a.dim()))); });
  return keepdim ? out.unsqueeze(d) : out;
}

Tensor arg_reduce_all(ArgReduceOp op, const char* name, const Tensor& a) {
  check_defined(name, a);
  check_numeric(name, a);
  FG_CHECK(a.numel() > 0, name, ": cannot reduce over zero elements, got ", describe(a));
  Tensor out = empty({}, {.dtype = DType::Int64, .device = a.device()});
  run_on_device(name, out, [&] { cpu::arg_reduce(op, out, a); });
  return out;
}

Tensor arg_reduce_dim(ArgReduceOp op, const char* name, const Tensor& a, std::int64_t dim,
                      bool keepdim) {
  check_defined(name, a);
  check_numeric(name, a);
  const std::int64_t d = normalize_dim(dim, a.dim(), name);
  Shape shape = a.shape();
  shape.erase(shape.begin() + d);
  FG_CHECK(a.size(d) > 0 || numel(shape) == 0, name,
           ": cannot reduce over zero elements (dimension ", dim, " of ", describe(a), ")");
  Tensor out = empty(shape, {.dtype = DType::Int64, .device = a.device()});
  run_on_device(name, out, [&] { cpu::arg_reduce(op, out, a.permute(move_to_end(d, a.dim()))); });
  return keepdim ? out.unsqueeze(d) : out;
}

Tensor softmax_impl(const char* name, const Tensor& a, std::int64_t dim, bool log_space) {
  check_defined(name, a);
  check_floating(name, a);
  const std::int64_t d = normalize_dim(dim, a.dim(), name);
  const std::vector<std::int64_t> perm = move_to_end(d, a.dim());
  Tensor out = empty(a.shape(), a.options());
  run_on_device(name, out, [&] { cpu::softmax(out.permute(perm), a.permute(perm), log_space); });
  return out;
}

}  // namespace

// ---- Elementwise binary ------------------------------------------------------------------

Tensor add(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Add, "add", a, b); }
Tensor sub(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Sub, "sub", a, b); }
Tensor mul(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Mul, "mul", a, b); }
Tensor div(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Div, "div", a, b); }
Tensor maximum(const Tensor& a, const Tensor& b) {
  return binary(BinaryOp::Maximum, "maximum", a, b);
}
Tensor minimum(const Tensor& a, const Tensor& b) {
  return binary(BinaryOp::Minimum, "minimum", a, b);
}
Tensor pow(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Pow, "pow", a, b); }

Tensor add(const Tensor& a, double b) { return add(a, scalar_like(b, a)); }
Tensor sub(const Tensor& a, double b) { return sub(a, scalar_like(b, a)); }
Tensor mul(const Tensor& a, double b) { return mul(a, scalar_like(b, a)); }
Tensor div(const Tensor& a, double b) { return div(a, scalar_like(b, a)); }
Tensor pow(const Tensor& a, double b) { return pow(a, scalar_like(b, a)); }

const Tensor& Tensor::add_(const Tensor& other) const {
  return binary_inplace(BinaryOp::Add, "add_", *this, other);
}
const Tensor& Tensor::sub_(const Tensor& other) const {
  return binary_inplace(BinaryOp::Sub, "sub_", *this, other);
}
const Tensor& Tensor::mul_(const Tensor& other) const {
  return binary_inplace(BinaryOp::Mul, "mul_", *this, other);
}
const Tensor& Tensor::div_(const Tensor& other) const {
  return binary_inplace(BinaryOp::Div, "div_", *this, other);
}
const Tensor& Tensor::add_(double other) const { return add_(scalar_like(other, *this)); }
const Tensor& Tensor::sub_(double other) const { return sub_(scalar_like(other, *this)); }
const Tensor& Tensor::mul_(double other) const { return mul_(scalar_like(other, *this)); }
const Tensor& Tensor::div_(double other) const { return div_(scalar_like(other, *this)); }

// ---- Comparisons -------------------------------------------------------------------------

Tensor eq(const Tensor& a, const Tensor& b) { return compare(CompareOp::Eq, "eq", a, b); }
Tensor ne(const Tensor& a, const Tensor& b) { return compare(CompareOp::Ne, "ne", a, b); }
Tensor lt(const Tensor& a, const Tensor& b) { return compare(CompareOp::Lt, "lt", a, b); }
Tensor le(const Tensor& a, const Tensor& b) { return compare(CompareOp::Le, "le", a, b); }
Tensor gt(const Tensor& a, const Tensor& b) { return compare(CompareOp::Gt, "gt", a, b); }
Tensor ge(const Tensor& a, const Tensor& b) { return compare(CompareOp::Ge, "ge", a, b); }

// ---- Elementwise unary -------------------------------------------------------------------

Tensor neg(const Tensor& a) { return unary(UnaryOp::Neg, "neg", a); }
Tensor abs(const Tensor& a) { return unary(UnaryOp::Abs, "abs", a); }
Tensor relu(const Tensor& a) { return unary(UnaryOp::Relu, "relu", a); }
Tensor exp(const Tensor& a) { return unary(UnaryOp::Exp, "exp", a); }
Tensor log(const Tensor& a) { return unary(UnaryOp::Log, "log", a); }
Tensor sqrt(const Tensor& a) { return unary(UnaryOp::Sqrt, "sqrt", a); }
Tensor tanh(const Tensor& a) { return unary(UnaryOp::Tanh, "tanh", a); }
Tensor sigmoid(const Tensor& a) { return unary(UnaryOp::Sigmoid, "sigmoid", a); }

// ---- Reductions --------------------------------------------------------------------------

Tensor sum(const Tensor& a) { return reduce_all(ReduceOp::Sum, "sum", a); }
Tensor sum(const Tensor& a, std::int64_t dim, bool keepdim) {
  return reduce_dim(ReduceOp::Sum, "sum", a, dim, keepdim);
}
Tensor mean(const Tensor& a) { return reduce_all(ReduceOp::Mean, "mean", a); }
Tensor mean(const Tensor& a, std::int64_t dim, bool keepdim) {
  return reduce_dim(ReduceOp::Mean, "mean", a, dim, keepdim);
}
Tensor max(const Tensor& a) { return reduce_all(ReduceOp::Max, "max", a); }
Tensor max(const Tensor& a, std::int64_t dim, bool keepdim) {
  return reduce_dim(ReduceOp::Max, "max", a, dim, keepdim);
}
Tensor min(const Tensor& a) { return reduce_all(ReduceOp::Min, "min", a); }
Tensor min(const Tensor& a, std::int64_t dim, bool keepdim) {
  return reduce_dim(ReduceOp::Min, "min", a, dim, keepdim);
}
Tensor argmax(const Tensor& a) { return arg_reduce_all(ArgReduceOp::Argmax, "argmax", a); }
Tensor argmax(const Tensor& a, std::int64_t dim, bool keepdim) {
  return arg_reduce_dim(ArgReduceOp::Argmax, "argmax", a, dim, keepdim);
}
Tensor argmin(const Tensor& a) { return arg_reduce_all(ArgReduceOp::Argmin, "argmin", a); }
Tensor argmin(const Tensor& a, std::int64_t dim, bool keepdim) {
  return arg_reduce_dim(ArgReduceOp::Argmin, "argmin", a, dim, keepdim);
}

// ---- Linear algebra ----------------------------------------------------------------------

Tensor matmul(const Tensor& a, const Tensor& b) {
  check_same_device_and_dtype("matmul", a, b);
  check_numeric("matmul", a);
  FG_CHECK(a.dim() >= 2 && b.dim() >= 2, "matmul: expected tensors with at least 2 dimensions, ",
           "got ", describe(a), " and ", describe(b));
  const std::int64_t m = a.size(-2);
  const std::int64_t k = a.size(-1);
  const std::int64_t n = b.size(-1);
  FG_CHECK(b.size(-2) == k, "matmul: inner dimensions differ, got ", describe(a), " and ",
           describe(b));
  const Shape batch = broadcast_shapes(Shape(a.shape().begin(), a.shape().end() - 2),
                                       Shape(b.shape().begin(), b.shape().end() - 2), "matmul");
  auto with_matrix = [&batch](std::int64_t rows, std::int64_t cols) {
    Shape shape = batch;
    shape.push_back(rows);
    shape.push_back(cols);
    return shape;
  };
  Tensor out = empty(with_matrix(m, n), a.options());
  run_on_device("matmul", out, [&] {
    cpu::matmul(out, a.expand(with_matrix(m, k)), b.expand(with_matrix(k, n)));
  });
  return out;
}

// ---- Normalization -----------------------------------------------------------------------

Tensor softmax(const Tensor& a, std::int64_t dim) { return softmax_impl("softmax", a, dim, false); }
Tensor log_softmax(const Tensor& a, std::int64_t dim) {
  return softmax_impl("log_softmax", a, dim, true);
}

// ---- Comparison helpers ------------------------------------------------------------------

bool allclose(const Tensor& a, const Tensor& b, double rtol, double atol) {
  check_defined("allclose", a);
  check_defined("allclose", b);
  if (a.shape() != b.shape()) return false;
  const std::vector<double> x = a.cpu().to(DType::Float64).to_vector<double>();
  const std::vector<double> y = b.cpu().to(DType::Float64).to_vector<double>();
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (x[i] == y[i]) continue;  // also covers equal infinities
    if (!std::isfinite(x[i]) || !std::isfinite(y[i])) return false;
    if (std::abs(x[i] - y[i]) > atol + rtol * std::abs(y[i])) return false;
  }
  return true;
}

bool equal(const Tensor& a, const Tensor& b) {
  check_defined("equal", a);
  check_defined("equal", b);
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) return false;
  return dispatch_all(a.dtype(), "equal",
                      [&]<typename T>() { return a.to_vector<T>() == b.to_vector<T>(); });
}

// ---- Operators ---------------------------------------------------------------------------

Tensor operator-(const Tensor& a) { return neg(a); }
Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }
Tensor operator+(const Tensor& a, double b) { return add(a, b); }
Tensor operator-(const Tensor& a, double b) { return sub(a, b); }
Tensor operator*(const Tensor& a, double b) { return mul(a, b); }
Tensor operator/(const Tensor& a, double b) { return div(a, b); }
Tensor operator+(double a, const Tensor& b) { return add(scalar_like(a, b), b); }
Tensor operator-(double a, const Tensor& b) { return sub(scalar_like(a, b), b); }
Tensor operator*(double a, const Tensor& b) { return mul(scalar_like(a, b), b); }
Tensor operator/(double a, const Tensor& b) { return div(scalar_like(a, b), b); }

Tensor& operator+=(Tensor& a, const Tensor& b) {
  a.add_(b);
  return a;
}
Tensor& operator-=(Tensor& a, const Tensor& b) {
  a.sub_(b);
  return a;
}
Tensor& operator*=(Tensor& a, const Tensor& b) {
  a.mul_(b);
  return a;
}
Tensor& operator/=(Tensor& a, const Tensor& b) {
  a.div_(b);
  return a;
}
Tensor& operator+=(Tensor& a, double b) {
  a.add_(b);
  return a;
}
Tensor& operator-=(Tensor& a, double b) {
  a.sub_(b);
  return a;
}
Tensor& operator*=(Tensor& a, double b) {
  a.mul_(b);
  return a;
}
Tensor& operator/=(Tensor& a, double b) {
  a.div_(b);
  return a;
}

}  // namespace flashgrad
