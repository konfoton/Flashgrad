#include "flashgrad/core/dtype.h"

namespace flashgrad {

std::string_view dtype_name(DType dtype) {
  switch (dtype) {
    case DType::Bool:
      return "bool";
    case DType::Int32:
      return "int32";
    case DType::Int64:
      return "int64";
    case DType::Float32:
      return "float32";
    case DType::Float64:
      return "float64";
  }
  return "unknown";
}

std::ostream& operator<<(std::ostream& os, DType dtype) { return os << dtype_name(dtype); }

}  // namespace flashgrad
