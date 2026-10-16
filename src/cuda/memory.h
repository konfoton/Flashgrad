#pragma once

#include <cstddef>

#include "flashgrad/core/allocator.h"

// Raw CUDA memory management used by Storage and Tensor::copy_. Kept free of CUDA headers
// so core code doesn't depend on cuda_runtime.h.
namespace flashgrad::cuda::detail {

// cudaMalloc on `device`; the returned pointer is released with cudaFree.
DataPtr allocate(std::size_t nbytes, int device);

void copy_host_to_device(void* dst, int dst_device, const void* src, std::size_t nbytes);
void copy_device_to_host(void* dst, const void* src, int src_device, std::size_t nbytes);
void copy_device_to_device(void* dst, int dst_device, const void* src, int src_device,
                           std::size_t nbytes);

}  // namespace flashgrad::cuda::detail
