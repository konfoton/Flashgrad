#pragma once

#include <cstdint>
#include <ostream>

#include "flashgrad/core/error.h"

namespace flashgrad {

enum class DeviceType : std::uint8_t { CPU, CUDA };

// A device a tensor lives on: the CPU, or a CUDA device identified by its index.
class Device {
 public:
  Device() = default;  // CPU
  Device(DeviceType type, int index = 0) : type_(type), index_(index) {
    FG_CHECK(index >= 0, "Device: negative device index ", index);
    FG_CHECK(type == DeviceType::CUDA || index == 0, "Device: CPU device index must be 0, got ",
             index);
  }

  static Device cpu() { return Device(); }
  static Device cuda(int index = 0) { return Device(DeviceType::CUDA, index); }

  DeviceType type() const noexcept { return type_; }
  int index() const noexcept { return index_; }
  bool is_cpu() const noexcept { return type_ == DeviceType::CPU; }
  bool is_cuda() const noexcept { return type_ == DeviceType::CUDA; }

  friend bool operator==(const Device&, const Device&) = default;

 private:
  DeviceType type_ = DeviceType::CPU;
  int index_ = 0;
};

std::ostream& operator<<(std::ostream& os, DeviceType type);

// "cpu" or "cuda:<index>".
std::ostream& operator<<(std::ostream& os, const Device& device);

}  // namespace flashgrad
