#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <optional>

#include "flashgrad/core/device.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/storage.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/allocator.h"
#include "flashgrad/cuda/runtime.h"

namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

#define SKIP_IF_NO_CACHING() \
  if (!flashgrad::cuda::caching_enabled()) GTEST_SKIP() << "FLASHGRAD_NO_CACHING is set"

namespace fg = flashgrad;
using fg::Storage;
namespace fgc = flashgrad::cuda;

constexpr std::size_t kMiB = std::size_t{1} << 20;

Storage gpu_storage(std::size_t nbytes) { return Storage(nbytes, fg::Device::cuda()); }

// Every test starts from an empty cache, so blocks cached by other tests can't be picked.
class CachingAllocator : public ::testing::Test {
 protected:
  void SetUp() override {
    SKIP_IF_NO_CUDA();
    fgc::empty_cache();
  }
};

TEST_F(CachingAllocator, ReusesAFreedBlock) {
  SKIP_IF_NO_CACHING();
  void* first = gpu_storage(100000).data();  // freed at the end of the statement
  const fgc::MemoryStats before = fgc::memory_stats();
  const Storage second = gpu_storage(100000);
  const fgc::MemoryStats after = fgc::memory_stats();
  EXPECT_EQ(second.data(), first);
  EXPECT_EQ(after.num_cache_hits, before.num_cache_hits + 1);
  EXPECT_EQ(after.num_cuda_mallocs, before.num_cuda_mallocs);
}

TEST_F(CachingAllocator, ReusesTheMostRecentlyFreedBlockOfASize) {
  SKIP_IF_NO_CACHING();
  std::optional<Storage> a = gpu_storage(4096);
  std::optional<Storage> b = gpu_storage(4096);
  void* b_ptr = b->data();
  a.reset();
  b.reset();
  EXPECT_EQ(gpu_storage(4096).data(), b_ptr);
}

TEST_F(CachingAllocator, PicksTheSmallestBlockThatFits) {
  SKIP_IF_NO_CACHING();
  std::optional<Storage> large = gpu_storage(3 * kMiB);
  std::optional<Storage> fits = gpu_storage(2 * kMiB);
  std::optional<Storage> small = gpu_storage(kMiB);
  void* fits_ptr = fits->data();
  large.reset();
  fits.reset();
  small.reset();
  EXPECT_EQ(gpu_storage(kMiB + 1).data(), fits_ptr);
}

TEST_F(CachingAllocator, SmallRequestDoesNotTakeALargeBlock) {
  SKIP_IF_NO_CACHING();
  void* large = gpu_storage(64 * kMiB).data();
  const fgc::MemoryStats before = fgc::memory_stats();
  const Storage small = gpu_storage(kMiB);
  EXPECT_NE(small.data(), large);
  EXPECT_EQ(fgc::memory_stats().num_cuda_mallocs, before.num_cuda_mallocs + 1);
  // At most min(request, 16 MiB) may be wasted: 64 MiB serves 50 MiB but not 40 MiB.
  EXPECT_NE(gpu_storage(40 * kMiB).data(), large);
  const fgc::MemoryStats cached = fgc::memory_stats();
  {
    const Storage reused = gpu_storage(50 * kMiB);
    EXPECT_EQ(reused.data(), large);
    // The whole 64 MiB block counts as allocated, and goes back to the cache as 64 MiB.
    EXPECT_EQ(fgc::memory_stats().allocated_bytes, cached.allocated_bytes + 64 * kMiB);
  }
  EXPECT_EQ(gpu_storage(64 * kMiB).data(), large);
}

TEST_F(CachingAllocator, RoundsSizesUpTo512Bytes) {
  const std::size_t base = fgc::memory_stats().allocated_bytes;
  const Storage one = gpu_storage(1);
  EXPECT_EQ(fgc::memory_stats().allocated_bytes, base + 512);
  const Storage more = gpu_storage(513);
  EXPECT_EQ(fgc::memory_stats().allocated_bytes, base + 512 + 1024);
  EXPECT_EQ(one.nbytes(), 1u);  // Storage keeps the requested size
}

TEST_F(CachingAllocator, TracksAllocatedReservedAndPeakBytes) {
  const fgc::MemoryStats start = fgc::memory_stats();
  EXPECT_EQ(start.reserved_bytes, start.allocated_bytes);  // nothing cached
  fgc::reset_peak_stats();
  {
    const Storage s = gpu_storage(8 * kMiB);
    const fgc::MemoryStats in_use = fgc::memory_stats();
    EXPECT_EQ(in_use.allocated_bytes, start.allocated_bytes + 8 * kMiB);
    EXPECT_EQ(in_use.reserved_bytes, start.reserved_bytes + 8 * kMiB);
  }
  const fgc::MemoryStats freed = fgc::memory_stats();
  EXPECT_EQ(freed.allocated_bytes, start.allocated_bytes);
  EXPECT_EQ(freed.peak_allocated_bytes, start.allocated_bytes + 8 * kMiB);
  EXPECT_EQ(freed.peak_reserved_bytes, start.reserved_bytes + 8 * kMiB);
  // Cached, or returned to CUDA right away without caching.
  EXPECT_EQ(freed.reserved_bytes, start.reserved_bytes + (fgc::caching_enabled() ? 8 * kMiB : 0));

  fgc::empty_cache();
  EXPECT_EQ(fgc::memory_stats().reserved_bytes, start.reserved_bytes);
  fgc::reset_peak_stats();
  EXPECT_EQ(fgc::memory_stats().peak_allocated_bytes, start.allocated_bytes);
}

TEST_F(CachingAllocator, ZeroBytesAllocateNothing) {
  const fgc::MemoryStats before = fgc::memory_stats();
  const Storage empty = gpu_storage(0);
  EXPECT_EQ(empty.data(), nullptr);
  const fgc::MemoryStats after = fgc::memory_stats();
  EXPECT_EQ(after.allocated_bytes, before.allocated_bytes);
  EXPECT_EQ(after.num_cuda_mallocs, before.num_cuda_mallocs);
  EXPECT_EQ(after.num_cache_hits, before.num_cache_hits);
}

TEST_F(CachingAllocator, RetriesAfterEmptyingTheCacheWhenOutOfMemory) {
  SKIP_IF_NO_CACHING();
  std::size_t free_bytes = 0, total_bytes = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
  // Cache a block of 60% of the free memory; a second, slightly larger one doesn't fit next
  // to it and is too large to reuse it.
  const std::size_t size = free_bytes / 10 * 6;
  gpu_storage(size);
  const fgc::MemoryStats before = fgc::memory_stats();
  const Storage second = gpu_storage(size + 64 * kMiB);
  EXPECT_NE(second.data(), nullptr);
  EXPECT_EQ(fgc::memory_stats().num_oom_retries, before.num_oom_retries + 1);
}

TEST_F(CachingAllocator, OutOfMemoryThrowsAndLeavesCudaUsable) {
  EXPECT_THROW(gpu_storage(std::size_t{1} << 50), fg::Error);  // 1 PiB
  // The failed cudaMalloc must not leave an error behind for the next kernel check.
  const fg::Tensor t = fg::ones({1000}, {.device = fg::Device::cuda()});
  EXPECT_FLOAT_EQ(fg::sum(t, 0).item(), 1000.0f);
}

TEST_F(CachingAllocator, RejectsInvalidDevices) {
  EXPECT_THROW(Storage(16, fg::Device::cuda(fgc::device_count())), fg::Error);
  EXPECT_THROW(Storage(0, fg::Device::cuda(fgc::device_count())), fg::Error);
  EXPECT_THROW(fgc::memory_stats(-1), fg::Error);
}

}  // namespace
