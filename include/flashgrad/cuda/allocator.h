#pragma once

#include <cstddef>
#include <cstdint>

// CUDA memory goes through a caching allocator: freed blocks are kept and handed out again
// instead of being returned with cudaFree, which waits for the whole device. Sizes are rounded
// up to multiples of kBlockRounding.
//
// The allocator assumes that all work runs on one stream. A block freed while a queued kernel
// still reads it can only go to a tensor used by a later kernel on that stream, which runs
// after the reader.
//
// Setting the environment variable FLASHGRAD_NO_CACHING=1 (read once, at the first CUDA
// allocation) turns caching off: every free goes straight to cudaFree. Use it with
// compute-sanitizer, which can't see out-of-bounds accesses or use-after-free inside cached
// blocks.
namespace flashgrad::cuda {

inline constexpr std::size_t kBlockRounding = 512;

struct MemoryStats {
  std::size_t allocated_bytes = 0;  // in blocks currently used by tensors (whole blocks)
  std::size_t reserved_bytes = 0;   // taken from cudaMalloc: allocated + cached
  std::size_t peak_allocated_bytes = 0;
  std::size_t peak_reserved_bytes = 0;
  std::int64_t num_cuda_mallocs = 0;
  std::int64_t num_cache_hits = 0;
  std::int64_t num_oom_retries = 0;  // cudaMalloc failures retried after emptying the cache
};

// Statistics since the start of the program (peaks since the last reset_peak_stats).
MemoryStats memory_stats(int device = 0);
void reset_peak_stats(int device = 0);

// Returns every cached (unused) block on every device to CUDA. Waits for the devices.
void empty_cache();

// False when FLASHGRAD_NO_CACHING is set.
bool caching_enabled();

}  // namespace flashgrad::cuda
