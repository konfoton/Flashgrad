#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string_view>

#include "flashgrad/core/error.h"

namespace flashgrad {

// Element type of a tensor. Reduced-precision types (float16, bfloat16) come in Phase 8.
enum class DType : std::uint8_t { Bool, Int32, Int64, Float32, Float64 };

constexpr std::size_t element_size(DType dtype) {
  switch (dtype) {
    case DType::Bool:
      return sizeof(bool);
    case DType::Int32:
      return sizeof(std::int32_t);
    case DType::Int64:
      return sizeof(std::int64_t);
    case DType::Float32:
      return sizeof(float);
    case DType::Float64:
      return sizeof(double);
  }
  return 0;
}

constexpr bool is_floating(DType dtype) {
  return dtype == DType::Float32 || dtype == DType::Float64;
}

// "bool", "int32", "int64", "float32", "float64".
std::string_view dtype_name(DType dtype);

std::ostream& operator<<(std::ostream& os, DType dtype);

// Maps a C++ type to its DType: dtype_of<float> == DType::Float32.
template <typename T>
struct DTypeOf;
template <>
struct DTypeOf<bool> {
  static constexpr DType value = DType::Bool;
};
template <>
struct DTypeOf<std::int32_t> {
  static constexpr DType value = DType::Int32;
};
template <>
struct DTypeOf<std::int64_t> {
  static constexpr DType value = DType::Int64;
};
template <>
struct DTypeOf<float> {
  static constexpr DType value = DType::Float32;
};
template <>
struct DTypeOf<double> {
  static constexpr DType value = DType::Float64;
};

template <typename T>
inline constexpr DType dtype_of = DTypeOf<T>::value;

// Runtime -> compile-time dtype dispatch. Calls `f.template operator()<T>()` with T the C++
// type stored by `dtype`, and returns its result:
//
//   dispatch_all(t.dtype(), "fill", [&]<typename T>() { ... });
//
// dispatch_numeric rejects bool; dispatch_floating accepts only float32 and float64.
// `op` names the calling operation in the error message.
template <typename F>
decltype(auto) dispatch_all(DType dtype, const char* op, F&& f) {
  switch (dtype) {
    case DType::Bool:
      return f.template operator()<bool>();
    case DType::Int32:
      return f.template operator()<std::int32_t>();
    case DType::Int64:
      return f.template operator()<std::int64_t>();
    case DType::Float32:
      return f.template operator()<float>();
    case DType::Float64:
      return f.template operator()<double>();
  }
  FG_THROW(op, ": unknown dtype ", static_cast<int>(dtype));
}

template <typename F>
decltype(auto) dispatch_numeric(DType dtype, const char* op, F&& f) {
  switch (dtype) {
    case DType::Int32:
      return f.template operator()<std::int32_t>();
    case DType::Int64:
      return f.template operator()<std::int64_t>();
    case DType::Float32:
      return f.template operator()<float>();
    case DType::Float64:
      return f.template operator()<double>();
    case DType::Bool:
      break;
  }
  FG_THROW(op, ": not supported for dtype ", dtype);
}

template <typename F>
decltype(auto) dispatch_floating(DType dtype, const char* op, F&& f) {
  switch (dtype) {
    case DType::Float32:
      return f.template operator()<float>();
    case DType::Float64:
      return f.template operator()<double>();
    case DType::Bool:
    case DType::Int32:
    case DType::Int64:
      break;
  }
  FG_THROW(op, ": expected a floating-point dtype, got ", dtype);
}

}  // namespace flashgrad
