#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string_view>

namespace flashgrad {

// Element type of a tensor. Only float32 is supported, so every tensor stores `float`
// elements; reduced-precision types (float16, bfloat16) are planned for Phase 8.
enum class DType : std::uint8_t { Float32 };

constexpr std::size_t element_size(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return sizeof(float);
  }
  return 0;
}

// "float32".
std::string_view dtype_name(DType dtype);

std::ostream& operator<<(std::ostream& os, DType dtype);

}  // namespace flashgrad
