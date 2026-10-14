#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"

// CUDA tensors in Phase 1: allocation and host <-> device transfers. Compute ops on CUDA
// tensors arrive in Phase 2.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

using flashgrad::Device;
using flashgrad::DType;
using flashgrad::Shape;
using flashgrad::Tensor;

Tensor random_tensor(const Shape& shape, std::uint64_t seed, DType dtype = DType::Float32) {
  flashgrad::Generator gen(seed);
  return flashgrad::randn(shape, {.dtype = dtype}, &gen);
}

TEST(CudaTensor, InvalidDeviceThrows) {
  // Works with or without a GPU: index == device_count() is never valid.
  EXPECT_THROW(flashgrad::empty({2}, {.device = Device::cuda(flashgrad::cuda::device_count())}),
               flashgrad::Error);
}

TEST(CudaTensor, EmptyAllocatesOnDevice) {
  SKIP_IF_NO_CUDA();
  const Tensor t = flashgrad::empty({4, 5}, {.device = Device::cuda()});
  EXPECT_EQ(t.device(), Device::cuda(0));
  EXPECT_EQ(t.storage().device(), Device::cuda(0));
  EXPECT_EQ(t.storage().nbytes(), 80u);
  EXPECT_NE(t.data_ptr(), nullptr);
  EXPECT_TRUE(t.is_contiguous());
}

class CudaRoundTrip : public ::testing::TestWithParam<std::int64_t> {};

TEST_P(CudaRoundTrip, CpuToCudaToCpuIsExact) {
  SKIP_IF_NO_CUDA();
  const std::int64_t n = GetParam();
  for (const DType dtype : {DType::Float32, DType::Float64}) {
    const Tensor host = random_tensor({n}, 1, dtype);
    const Tensor dev = host.cuda();
    EXPECT_TRUE(dev.device().is_cuda());
    EXPECT_FALSE(dev.storage().is_same(host.storage()));
    const Tensor back = dev.cpu();
    EXPECT_TRUE(back.device().is_cpu());
    EXPECT_TRUE(flashgrad::equal(back, host)) << "n=" << n << " dtype=" << dtype;
  }
}

// Empty, single element, non-multiples of the block size, and a larger buffer.
INSTANTIATE_TEST_SUITE_P(Sizes, CudaRoundTrip, ::testing::Values(0, 1, 255, 257, 1'000'003));

TEST(CudaTensor, RoundTripAllDtypes) {
  SKIP_IF_NO_CUDA();
  const Tensor i32 = flashgrad::tensor({1, -2, 3});
  const Tensor i64 = flashgrad::tensor({std::int64_t{1} << 40, std::int64_t{-5}});
  const Tensor b = flashgrad::tensor({true, false, true});
  EXPECT_TRUE(flashgrad::equal(i32.cuda().cpu(), i32));
  EXPECT_TRUE(flashgrad::equal(i64.cuda().cpu(), i64));
  EXPECT_TRUE(flashgrad::equal(b.cuda().cpu(), b));
}

TEST(CudaTensor, NonContiguousHostSourceIsPackedOnUpload) {
  SKIP_IF_NO_CUDA();
  const Tensor host = random_tensor({6, 8}, 2).t().slice(0, 1, 8, 2);  // [4, 6], strided
  ASSERT_FALSE(host.is_contiguous());
  const Tensor dev = host.cuda();
  EXPECT_TRUE(dev.is_contiguous());
  EXPECT_TRUE(flashgrad::equal(dev.cpu(), host.contiguous()));
}

TEST(CudaTensor, NonContiguousDeviceViewsDownloadCorrectly) {
  SKIP_IF_NO_CUDA();
  const Tensor host = random_tensor({5, 6, 7}, 3);
  const Tensor dev = host.cuda();
  // Views of a CUDA tensor are pure metadata; downloading must honour their strides.
  EXPECT_TRUE(flashgrad::equal(dev.t().cpu(), host.t().contiguous()));
  EXPECT_TRUE(flashgrad::equal(dev.permute({2, 0, 1}).cpu(), host.permute({2, 0, 1})));
  EXPECT_TRUE(flashgrad::equal(dev.slice(1, 1, 6, 2).cpu(), host.slice(1, 1, 6, 2)));
  EXPECT_TRUE(flashgrad::equal(dev[4][5].cpu(), host[4][5]));
  EXPECT_TRUE(flashgrad::equal(dev.slice(2, 3, 4).cpu(), host.slice(2, 3, 4)));
  EXPECT_EQ(dev[2][3][4].item<float>(), host[2][3][4].item<float>());

  const Tensor col = flashgrad::tensor({1.0f, 2.0f, 3.0f}, {3, 1}).cuda();
  EXPECT_EQ(col.expand({3, 4}).cpu().to_vector<float>(),
            (std::vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}));
}

TEST(CudaTensor, CopyIntoHostViewConvertsDtype) {
  SKIP_IF_NO_CUDA();
  const Tensor dev = flashgrad::tensor({1.5f, 2.5f, 3.5f}).cuda();
  const Tensor host = flashgrad::zeros({2, 3}, {.dtype = DType::Float64});
  host.copy_(dev);  // CUDA -> CPU with broadcasting and float32 -> float64
  EXPECT_EQ(host.to_vector<double>(), (std::vector<double>{1.5, 2.5, 3.5, 1.5, 2.5, 3.5}));
}

TEST(CudaTensor, CopyFromHostConvertsDtype) {
  SKIP_IF_NO_CUDA();
  const Tensor dev = flashgrad::empty({3}, {.dtype = DType::Int32, .device = Device::cuda()});
  dev.copy_(flashgrad::tensor({1.9, -2.9, 3.0}));  // float64 host -> int32 device
  EXPECT_EQ(dev.cpu().to_vector<std::int32_t>(), (std::vector<std::int32_t>{1, -2, 3}));
}

TEST(CudaTensor, DeviceToDeviceCopy) {
  SKIP_IF_NO_CUDA();
  const Tensor host = random_tensor({100}, 4);
  const Tensor a = host.cuda();
  const Tensor b = a.clone();
  EXPECT_TRUE(b.device().is_cuda());
  EXPECT_FALSE(b.storage().is_same(a.storage()));
  EXPECT_TRUE(flashgrad::equal(b.cpu(), host));
  // A contiguous slice of a CUDA tensor can be copied on the device.
  const Tensor part = flashgrad::empty({10}, {.device = Device::cuda()});
  part.copy_(a.slice(0, 20, 30));
  EXPECT_TRUE(flashgrad::equal(part.cpu(), host.slice(0, 20, 30)));
}

TEST(CudaTensor, Factories) {
  SKIP_IF_NO_CUDA();
  const flashgrad::TensorOptions on_gpu{.dtype = DType::Float32, .device = Device::cuda()};
  EXPECT_EQ(flashgrad::zeros({2, 3}, on_gpu).cpu().to_vector<float>(), std::vector<float>(6, 0.0f));
  EXPECT_EQ(flashgrad::ones({4}, on_gpu).cpu().to_vector<float>(), std::vector<float>(4, 1.0f));
  EXPECT_EQ(flashgrad::full({2}, 7.0, on_gpu).cpu().to_vector<float>(), (std::vector<float>{7, 7}));
  EXPECT_EQ(flashgrad::arange(4.0, on_gpu).cpu().to_vector<float>(),
            (std::vector<float>{0, 1, 2, 3}));
  EXPECT_TRUE(flashgrad::equal(flashgrad::eye(3, on_gpu).cpu(), flashgrad::eye(3)));
  EXPECT_TRUE(
      flashgrad::equal(flashgrad::tensor({1, 2}, Device::cuda()).cpu(), flashgrad::tensor({1, 2})));
  EXPECT_EQ(flashgrad::zeros({0}, on_gpu).numel(), 0);
}

TEST(CudaTensor, RandomFactoriesMatchCpuForSameSeed) {
  SKIP_IF_NO_CUDA();
  flashgrad::Generator g1(5);
  flashgrad::Generator g2(5);
  const Tensor dev = flashgrad::randn({64}, {.device = Device::cuda()}, &g1);
  const Tensor host = flashgrad::randn({64}, {}, &g2);
  EXPECT_TRUE(flashgrad::equal(dev.cpu(), host));
}

TEST(CudaTensor, FillContiguous) {
  SKIP_IF_NO_CUDA();
  const Tensor dev = flashgrad::empty({5}, {.dtype = DType::Int64, .device = Device::cuda()});
  dev.fill_(3.0);
  EXPECT_EQ(dev.cpu().to_vector<std::int64_t>(), (std::vector<std::int64_t>(5, 3)));
  dev.zero_();
  EXPECT_EQ(dev.cpu().to_vector<std::int64_t>(), (std::vector<std::int64_t>(5, 0)));
}

TEST(CudaTensor, ItemAndPrinting) {
  SKIP_IF_NO_CUDA();
  const Tensor dev = flashgrad::tensor({1, 2, 3, 4}, {2, 2}, Device::cuda());
  EXPECT_EQ(dev[1][0].item<std::int32_t>(), 3);
  std::ostringstream oss;
  oss << dev;
  EXPECT_EQ(oss.str(), "tensor([[1, 2],\n        [3, 4]], dtype=int32, device=cuda:0)");
}

TEST(CudaTensor, ViewsKeepDevice) {
  SKIP_IF_NO_CUDA();
  const Tensor dev = flashgrad::zeros({2, 3}, {.device = Device::cuda()});
  EXPECT_EQ(dev.t().device(), Device::cuda());
  EXPECT_TRUE(dev.view({3, 2}).storage().is_same(dev.storage()));
  EXPECT_TRUE(dev.cuda().is_same(dev));  // already there: no copy
}

TEST(CudaTensor, ComputeOpsNotImplementedYet) {
  SKIP_IF_NO_CUDA();
  const Tensor dev = flashgrad::ones({3}, {.device = Device::cuda()});
  try {
    (void)(dev + dev);
    FAIL() << "expected an error";
  } catch (const flashgrad::Error& e) {
    const std::string msg = e.message();
    EXPECT_NE(msg.find("add"), std::string::npos) << msg;
    EXPECT_NE(msg.find("cuda:0"), std::string::npos) << msg;
  }
  EXPECT_THROW(flashgrad::sum(dev), flashgrad::Error);
  EXPECT_THROW(flashgrad::matmul(dev.view({1, 3}), dev.view({3, 1})), flashgrad::Error);
  // On-device dtype conversion and non-contiguous device-to-device copies need kernels.
  EXPECT_THROW(dev.to(DType::Float64), flashgrad::Error);
  EXPECT_THROW(flashgrad::zeros({3, 2}, {.device = Device::cuda()}).t().contiguous(),
               flashgrad::Error);
}

TEST(CudaTensor, MixedDevicesThrow) {
  SKIP_IF_NO_CUDA();
  const Tensor dev = flashgrad::ones({3}, {.device = Device::cuda()});
  const Tensor host = flashgrad::ones({3});
  EXPECT_THROW(dev + host, flashgrad::Error);
  EXPECT_THROW(flashgrad::matmul(host.view({1, 3}), dev.view({3, 1})), flashgrad::Error);
}

}  // namespace
