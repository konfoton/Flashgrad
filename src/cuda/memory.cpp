#include "cuda/memory.h"

#include <cuda_runtime.h>

#include "flashgrad/cuda/check.h"
#include "flashgrad/cuda/runtime.h"

namespace flashgrad::cuda::detail {

namespace {

// Deleter for cudaMalloc'ed memory. Must not throw, so the error is ignored.
void cuda_free(void* ptr) { cudaFree(ptr); }

void check_device(const char* op, int device) {
  const int count = device_count();
  FG_CHECK(device >= 0 && device < count, op, ": invalid CUDA device cuda:", device, " (", count,
           " device(s) available)");
}

}  // namespace

DeviceGuard::DeviceGuard(int device) {
  FG_CUDA_CHECK(cudaGetDevice(&previous_));
  if (device != previous_) FG_CUDA_CHECK(cudaSetDevice(device));
}

DeviceGuard::~DeviceGuard() { cudaSetDevice(previous_); }

DataPtr allocate(std::size_t nbytes, int device) {
  check_device("allocate", device);
  if (nbytes == 0) return DataPtr(nullptr, &cuda_free);
  DeviceGuard guard(device);
  void* ptr = nullptr;
  FG_CUDA_CHECK(cudaMalloc(&ptr, nbytes));
  return DataPtr(ptr, &cuda_free);
}

void copy_host_to_device(void* dst, int dst_device, const void* src, std::size_t nbytes) {
  if (nbytes == 0) return;
  DeviceGuard guard(dst_device);
  FG_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
}

void copy_device_to_host(void* dst, const void* src, int src_device, std::size_t nbytes) {
  if (nbytes == 0) return;
  DeviceGuard guard(src_device);
  FG_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost));
}

void copy_device_to_device(void* dst, int dst_device, const void* src, int src_device,
                           std::size_t nbytes) {
  if (nbytes == 0) return;
  if (dst_device == src_device) {
    DeviceGuard guard(dst_device);
    FG_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
  } else {
    FG_CUDA_CHECK(cudaMemcpyPeer(dst, dst_device, src, src_device, nbytes));
  }
}

}  // namespace flashgrad::cuda::detail
