#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "flashgrad/core/device.h"
#include "flashgrad/core/dtype.h"
#include "flashgrad/core/error.h"

namespace {

using flashgrad::Device;
using flashgrad::DeviceType;
using flashgrad::DType;

template <typename T>
std::string str(const T& value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

TEST(DType, ElementSize) { EXPECT_EQ(flashgrad::element_size(DType::Float32), 4u); }

TEST(DType, NamesAndPrinting) {
  EXPECT_EQ(flashgrad::dtype_name(DType::Float32), "float32");
  EXPECT_EQ(str(DType::Float32), "float32");
}

TEST(Device, DefaultIsCpu) {
  const Device d;
  EXPECT_TRUE(d.is_cpu());
  EXPECT_EQ(d.type(), DeviceType::CPU);
  EXPECT_EQ(d.index(), 0);
  EXPECT_EQ(d, Device::cpu());
}

TEST(Device, CudaDevice) {
  const Device d = Device::cuda(1);
  EXPECT_TRUE(d.is_cuda());
  EXPECT_EQ(d.index(), 1);
  EXPECT_NE(d, Device::cuda(0));
  EXPECT_NE(d, Device::cpu());
}

TEST(Device, Printing) {
  EXPECT_EQ(str(Device::cpu()), "cpu");
  EXPECT_EQ(str(Device::cuda(2)), "cuda:2");
}

TEST(Device, InvalidIndexThrows) {
  EXPECT_THROW(Device::cuda(-1), flashgrad::Error);
  EXPECT_THROW(Device(DeviceType::CPU, 1), flashgrad::Error);
}

}  // namespace
