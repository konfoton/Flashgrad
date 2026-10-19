#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "cuda/memory.h"
#include "flashgrad/core/error.h"
#include "flashgrad/cuda/allocator.h"
#include "flashgrad/cuda/check.h"
#include "flashgrad/cuda/runtime.h"

// Caching allocator for device memory (see flashgrad/cuda/allocator.h). Free blocks are kept
// per device in a multimap sorted by size, so lower_bound finds the smallest cached block that
// is large enough (best fit). Blocks are never split or merged.
namespace flashgrad::cuda {

namespace {

// A cached block may be larger than the request by at most this much (and at most the
// request's own size), so a small tensor doesn't pin a large block.
constexpr std::size_t kMaxWaste = std::size_t{16} << 20;

std::size_t round_up(std::size_t nbytes) {
  return (nbytes + kBlockRounding - 1) / kBlockRounding * kBlockRounding;
}

struct Block {
  std::size_t size;
  int device;
};

struct DevicePool {
  std::multimap<std::size_t, void*> free_blocks;  // size -> ptr
  MemoryStats stats;
};

class CachingAllocator {
 public:
  CachingAllocator() {
    const char* env = std::getenv("FLASHGRAD_NO_CACHING");
    enabled_ = env == nullptr || std::strcmp(env, "0") == 0;
  }

  bool enabled() const { return enabled_; }

  void* allocate(std::size_t nbytes, int device) {
    std::size_t size = round_up(nbytes);
    std::lock_guard<std::mutex> lock(mutex_);
    DevicePool& pool = pool_for(device);
    void* ptr = take_cached(pool, size);  // may grow `size` to the cached block's size
    if (ptr != nullptr) {
      ++pool.stats.num_cache_hits;
    } else {
      ptr = cuda_malloc(pool, size, device);
      ++pool.stats.num_cuda_mallocs;
      pool.stats.reserved_bytes += size;
      pool.stats.peak_reserved_bytes =
          std::max(pool.stats.peak_reserved_bytes, pool.stats.reserved_bytes);
    }
    pool.stats.allocated_bytes += size;
    pool.stats.peak_allocated_bytes =
        std::max(pool.stats.peak_allocated_bytes, pool.stats.allocated_bytes);
    allocated_.emplace(ptr, Block{size, device});
    return ptr;
  }

  // Called from DataPtr's deleter, so it must not throw.
  void free(void* ptr) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = allocated_.find(ptr);
    if (it == allocated_.end()) return;  // not ours; can't happen
    const Block block = it->second;
    allocated_.erase(it);
    DevicePool& pool = pools_[static_cast<std::size_t>(block.device)];
    pool.stats.allocated_bytes -= block.size;
    if (enabled_) {
      // Inserted before blocks of the same size, so the most recently freed one is reused first.
      pool.free_blocks.emplace_hint(pool.free_blocks.lower_bound(block.size), block.size, ptr);
    } else {
      cudaFree(ptr);
      pool.stats.reserved_bytes -= block.size;
    }
  }

  void empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t d = 0; d < pools_.size(); ++d) release_cached(pools_[d], static_cast<int>(d));
  }

  MemoryStats stats(int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_for(device).stats;
  }

  void reset_peak_stats(int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    MemoryStats& stats = pool_for(device).stats;
    stats.peak_allocated_bytes = stats.allocated_bytes;
    stats.peak_reserved_bytes = stats.reserved_bytes;
  }

 private:
  DevicePool& pool_for(int device) {
    if (pools_.empty()) pools_.resize(static_cast<std::size_t>(device_count()));
    FG_CHECK(device >= 0 && static_cast<std::size_t>(device) < pools_.size(),
             "CUDA allocator: invalid device cuda:", device, " (", pools_.size(),
             " device(s) available)");
    return pools_[static_cast<std::size_t>(device)];
  }

  // The smallest cached block of at least `size` bytes, if it doesn't waste too much. Sets
  // `size` to the block's size: the block keeps it when it goes back to the cache.
  static void* take_cached(DevicePool& pool, std::size_t& size) {
    const auto it = pool.free_blocks.lower_bound(size);
    if (it == pool.free_blocks.end() || it->first - size > std::min(size, kMaxWaste)) {
      return nullptr;
    }
    size = it->first;
    void* ptr = it->second;
    pool.free_blocks.erase(it);
    return ptr;
  }

  // cudaMalloc; when the device is out of memory, returns the cached blocks and tries again.
  void* cuda_malloc(DevicePool& pool, std::size_t size, int device) {
    detail::DeviceGuard guard(device);
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err == cudaErrorMemoryAllocation && !pool.free_blocks.empty()) {
      (void)cudaGetLastError();  // clear it, so later checks don't report it again
      ++pool.stats.num_oom_retries;
      release_cached(pool, device);
      err = cudaMalloc(&ptr, size);
    }
    if (err == cudaErrorMemoryAllocation) {
      (void)cudaGetLastError();
      FG_THROW("CUDA out of memory: tried to allocate ", size, " bytes on cuda:", device, " (",
               pool.stats.allocated_bytes, " bytes allocated by tensors, ",
               pool.stats.reserved_bytes, " bytes reserved)");
    }
    FG_CUDA_CHECK(err);
    return ptr;
  }

  static void release_cached(DevicePool& pool, int device) {
    detail::DeviceGuard guard(device);
    for (const auto& [size, ptr] : pool.free_blocks) {
      FG_CUDA_CHECK(cudaFree(ptr));
      pool.stats.reserved_bytes -= size;
    }
    pool.free_blocks.clear();
  }

  std::mutex mutex_;
  bool enabled_ = true;
  std::vector<DevicePool> pools_;              // indexed by device
  std::unordered_map<void*, Block> allocated_;  // blocks in use, on every device
};

// Never destroyed: blocks may still be freed during static destruction, and the CUDA runtime
// may already be shut down by then. The OS reclaims the memory at exit.
CachingAllocator& allocator() {
  static auto* instance = new CachingAllocator();
  return *instance;
}

void cached_free(void* ptr) { allocator().free(ptr); }

}  // namespace

MemoryStats memory_stats(int device) { return allocator().stats(device); }

void reset_peak_stats(int device) { allocator().reset_peak_stats(device); }

void empty_cache() { allocator().empty_cache(); }

bool caching_enabled() { return allocator().enabled(); }

namespace detail {

DataPtr allocate(std::size_t nbytes, int device) {
  if (nbytes == 0) {
    const int count = device_count();
    FG_CHECK(device >= 0 && device < count, "allocate: invalid CUDA device cuda:", device, " (",
             count, " device(s) available)");
    return DataPtr(nullptr, &cached_free);
  }
  return DataPtr(allocator().allocate(nbytes, device), &cached_free);
}

}  // namespace detail

}  // namespace flashgrad::cuda
