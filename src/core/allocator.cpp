#include "flashgrad/core/allocator.h"

#include <new>

#include "cuda/memory.h"
#include "flashgrad/core/error.h"

namespace flashgrad {

namespace {

void cpu_free(void* ptr) { ::operator delete(ptr, std::align_val_t{kCpuAlignment}); }

Buffer cpu_allocate(std::size_t nbytes) {
  if (nbytes == 0) return Buffer(nullptr, &cpu_free);
  return Buffer(::operator new(nbytes, std::align_val_t{kCpuAlignment}), &cpu_free);
}

}  // namespace

Buffer allocate(std::size_t nbytes, Device device) {
  switch (device.type()) {
    case DeviceType::CPU:
      return cpu_allocate(nbytes);
    case DeviceType::CUDA:
      return cuda::detail::allocate(nbytes, device.index());
  }
  FG_THROW("allocate: unknown device ", device);
}

}  // namespace flashgrad
