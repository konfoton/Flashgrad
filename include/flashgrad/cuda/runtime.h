#pragma once

#include <cstddef>
#include <ostream>
#include <string>

namespace flashgrad::cuda {

struct DeviceProperties {
  int index = 0;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  std::size_t total_global_mem = 0;      // bytes
  std::size_t shared_mem_per_block = 0;  // bytes (default, without opt-in)
  std::size_t shared_mem_per_block_optin = 0;
  std::size_t shared_mem_per_sm = 0;
  int multiprocessor_count = 0;
  int max_threads_per_block = 0;
  int max_threads_per_sm = 0;
  int warp_size = 0;
  int regs_per_block = 0;
  int l2_cache_size = 0;  // bytes
  int memory_bus_width = 0;  // bits
};

// Number of visible CUDA devices. Returns 0 (never throws) when no driver
// or no GPU is available, so callers can use it to skip GPU code paths.
int device_count();

bool is_available();

int current_device();
void set_device(int index);
void synchronize();

DeviceProperties device_properties(int index);

std::ostream& operator<<(std::ostream& os, const DeviceProperties& props);

}  // namespace flashgrad::cuda
