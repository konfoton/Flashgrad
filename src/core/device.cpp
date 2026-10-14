#include "flashgrad/core/device.h"

namespace flashgrad {

std::ostream& operator<<(std::ostream& os, DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return os << "cpu";
    case DeviceType::CUDA:
      return os << "cuda";
  }
  return os << "unknown";
}

std::ostream& operator<<(std::ostream& os, const Device& device) {
  os << device.type();
  if (device.is_cuda()) os << ':' << device.index();
  return os;
}

}  // namespace flashgrad
