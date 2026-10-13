#include "flashgrad/cuda/runtime.h"

#include <cuda_runtime.h>

#include <iomanip>

#include "flashgrad/cuda/check.h"

namespace flashgrad::cuda {

int device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    // Clear the sticky error so later CUDA calls aren't affected.
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

bool is_available() { return device_count() > 0; }

int current_device() {
  int index = 0;
  FG_CUDA_CHECK(cudaGetDevice(&index));
  return index;
}

void set_device(int index) {
  const int count = device_count();
  FG_CHECK(index >= 0 && index < count, "set_device: invalid device index ", index,
           " (", count, " device(s) available)");
  FG_CUDA_CHECK(cudaSetDevice(index));
}

void synchronize() { FG_CUDA_CHECK(cudaDeviceSynchronize()); }

DeviceProperties device_properties(int index) {
  const int count = device_count();
  FG_CHECK(index >= 0 && index < count, "device_properties: invalid device index ",
           index, " (", count, " device(s) available)");

  cudaDeviceProp p{};
  FG_CUDA_CHECK(cudaGetDeviceProperties(&p, index));

  DeviceProperties out;
  out.index = index;
  out.name = p.name;
  out.compute_major = p.major;
  out.compute_minor = p.minor;
  out.total_global_mem = p.totalGlobalMem;
  out.shared_mem_per_block = p.sharedMemPerBlock;
  out.shared_mem_per_block_optin = p.sharedMemPerBlockOptin;
  out.shared_mem_per_sm = p.sharedMemPerMultiprocessor;
  out.multiprocessor_count = p.multiProcessorCount;
  out.max_threads_per_block = p.maxThreadsPerBlock;
  out.max_threads_per_sm = p.maxThreadsPerMultiProcessor;
  out.warp_size = p.warpSize;
  out.regs_per_block = p.regsPerBlock;
  out.l2_cache_size = p.l2CacheSize;
  out.memory_bus_width = p.memoryBusWidth;
  return out;
}

std::ostream& operator<<(std::ostream& os, const DeviceProperties& p) {
  constexpr double kKiB = 1024.0;
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  const auto flags = os.flags();
  const auto precision = os.precision();
  os << "Device " << p.index << ": " << p.name << "\n"
     << "  compute capability   : " << p.compute_major << "." << p.compute_minor << "\n"
     << std::fixed << std::setprecision(2)
     << "  global memory        : " << static_cast<double>(p.total_global_mem) / kGiB
     << " GiB\n"
     << std::setprecision(0)
     << "  SMs                  : " << p.multiprocessor_count << "\n"
     << "  max threads / block  : " << p.max_threads_per_block << "\n"
     << "  max threads / SM     : " << p.max_threads_per_sm << "\n"
     << "  warp size            : " << p.warp_size << "\n"
     << "  shared mem / block   : " << static_cast<double>(p.shared_mem_per_block) / kKiB
     << " KiB (opt-in " << static_cast<double>(p.shared_mem_per_block_optin) / kKiB
     << " KiB)\n"
     << "  shared mem / SM      : " << static_cast<double>(p.shared_mem_per_sm) / kKiB
     << " KiB\n"
     << "  registers / block    : " << p.regs_per_block << "\n"
     << "  L2 cache             : " << static_cast<double>(p.l2_cache_size) / kKiB
     << " KiB\n"
     << "  memory bus width     : " << p.memory_bus_width << " bit";
  os.flags(flags);
  os.precision(precision);
  return os;
}

}  // namespace flashgrad::cuda
