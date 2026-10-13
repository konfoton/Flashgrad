#include "flashgrad/cuda/runtime.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "flashgrad/cuda/check.h"

namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

TEST(CudaRuntime, DeviceCountIsNonNegative) { EXPECT_GE(flashgrad::cuda::device_count(), 0); }

TEST(CudaRuntime, DevicePropertiesAreSane) {
  SKIP_IF_NO_CUDA();
  const auto props = flashgrad::cuda::device_properties(0);
  EXPECT_FALSE(props.name.empty());
  EXPECT_GE(props.compute_major, 7);  // we target Volta and newer
  EXPECT_EQ(props.warp_size, 32);
  EXPECT_GT(props.multiprocessor_count, 0);
  EXPECT_GT(props.total_global_mem, 0u);
}

TEST(CudaRuntime, InvalidDeviceIndexThrows) {
  EXPECT_THROW(flashgrad::cuda::device_properties(-1), flashgrad::Error);
  EXPECT_THROW(flashgrad::cuda::set_device(flashgrad::cuda::device_count()), flashgrad::Error);
}

TEST(CudaRuntime, CudaCheckReportsErrorName) {
  SKIP_IF_NO_CUDA();
  try {
    // Asking for an absurd amount of memory must fail cleanly.
    void* p = nullptr;
    FG_CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(1) << 60));
    FAIL() << "FG_CUDA_CHECK did not throw";
  } catch (const flashgrad::Error& e) {
    EXPECT_NE(std::string(e.what()).find("cudaErrorMemoryAllocation"), std::string::npos)
        << e.what();
  }
  // Non-sticky error: the device must still be usable afterwards.
  (void)cudaGetLastError();
  EXPECT_NO_THROW(flashgrad::cuda::synchronize());
}

}  // namespace
