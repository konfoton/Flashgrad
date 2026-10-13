#include "flashgrad/cuda/smoke.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "flashgrad/cuda/check.h"

namespace flashgrad::cuda::smoke {

namespace {

__global__ void vector_add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ out, std::int64_t n) {
  // Grid-stride loop: correct for any n regardless of the launch size.
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}

// Minimal RAII device buffer; replaced by Storage + caching allocator later.
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) {
    if (bytes > 0) FG_CUDA_CHECK(cudaMalloc(&ptr_, bytes));
  }
  ~DeviceBuffer() { cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  float* data() const { return static_cast<float*>(ptr_); }

 private:
  void* ptr_ = nullptr;
};

}  // namespace

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
  FG_CHECK(a.size() == b.size(), "vector_add: size mismatch (", a.size(), " vs ", b.size(),
           ")");
  const std::size_t n = a.size();
  std::vector<float> out(n);
  if (n == 0) return out;

  const std::size_t bytes = n * sizeof(float);
  DeviceBuffer d_a(bytes), d_b(bytes), d_out(bytes);

  FG_CUDA_CHECK(cudaMemcpy(d_a.data(), a.data(), bytes, cudaMemcpyHostToDevice));
  FG_CUDA_CHECK(cudaMemcpy(d_b.data(), b.data(), bytes, cudaMemcpyHostToDevice));

  constexpr int kThreads = 256;
  constexpr std::size_t kMaxBlocks = 65535;
  const std::size_t blocks_needed = (n + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(blocks_needed < kMaxBlocks ? blocks_needed : kMaxBlocks);

  vector_add_kernel<<<blocks, kThreads>>>(d_a.data(), d_b.data(), d_out.data(),
                                          static_cast<std::int64_t>(n));
  FG_CUDA_KERNEL_CHECK();

  FG_CUDA_CHECK(cudaMemcpy(out.data(), d_out.data(), bytes, cudaMemcpyDeviceToHost));
  return out;
}

}  // namespace flashgrad::cuda::smoke
