#pragma once

#include <cstddef>
#include <memory>

#include "flashgrad/core/device.h"

namespace flashgrad {

// Owning pointer to device or host memory. The deleter knows how to release it, so storage
// code never calls free/cudaFree directly. CUDA memory is returned to the caching allocator
// (flashgrad/cuda/allocator.h).
using DataPtr = std::unique_ptr<void, void (*)(void*)>;

// Alignment of every CPU allocation (one cache line; enough for any SIMD load).
inline constexpr std::size_t kCpuAlignment = 64;

// Allocates `nbytes` of uninitialized memory on `device`. Zero bytes yield a null DataPtr.
DataPtr allocate(std::size_t nbytes, Device device);

}  // namespace flashgrad
