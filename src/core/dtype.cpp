#include "flashgrad/core/dtype.h"

namespace flashgrad {

std::string_view dtype_name(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return "float32";
  }
  return "unknown";
}

std::ostream& operator<<(std::ostream& os, DType dtype) { return os << dtype_name(dtype); }

}  // namespace flashgrad
