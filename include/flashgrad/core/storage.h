#pragma once

#include <cstddef>
#include <memory>

#include "flashgrad/core/allocator.h"
#include "flashgrad/core/device.h"

namespace flashgrad {

// A reference-counted, untyped buffer on one device. Copies of a Storage share the same
// memory; tensors that are views of each other share one Storage.
class Storage {
 public:
  Storage() = default;
  // Allocates `nbytes` of uninitialized memory on `device`.
  Storage(std::size_t nbytes, Device device);

  bool defined() const noexcept { return impl_ != nullptr; }
  void* data() const noexcept;
  std::size_t nbytes() const noexcept;
  Device device() const noexcept;

  // True when both handles refer to the same buffer.
  bool is_same(const Storage& other) const noexcept { return impl_ == other.impl_; }
  long use_count() const noexcept { return impl_.use_count(); }

 private:
  struct Impl {
    DataPtr data;
    std::size_t nbytes;
    Device device;
  };
  std::shared_ptr<Impl> impl_;
};

}  // namespace flashgrad
