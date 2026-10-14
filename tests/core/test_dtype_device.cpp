#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>

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

TEST(DType, ElementSizes) {
  EXPECT_EQ(flashgrad::element_size(DType::Bool), 1u);
  EXPECT_EQ(flashgrad::element_size(DType::Int32), 4u);
  EXPECT_EQ(flashgrad::element_size(DType::Int64), 8u);
  EXPECT_EQ(flashgrad::element_size(DType::Float32), 4u);
  EXPECT_EQ(flashgrad::element_size(DType::Float64), 8u);
}

TEST(DType, NamesAndPrinting) {
  EXPECT_EQ(flashgrad::dtype_name(DType::Bool), "bool");
  EXPECT_EQ(flashgrad::dtype_name(DType::Int64), "int64");
  EXPECT_EQ(str(DType::Float32), "float32");
}

TEST(DType, IsFloating) {
  EXPECT_TRUE(flashgrad::is_floating(DType::Float32));
  EXPECT_TRUE(flashgrad::is_floating(DType::Float64));
  EXPECT_FALSE(flashgrad::is_floating(DType::Int32));
  EXPECT_FALSE(flashgrad::is_floating(DType::Bool));
}

TEST(DType, DTypeOf) {
  static_assert(flashgrad::dtype_of<float> == DType::Float32);
  static_assert(flashgrad::dtype_of<double> == DType::Float64);
  static_assert(flashgrad::dtype_of<std::int32_t> == DType::Int32);
  static_assert(flashgrad::dtype_of<std::int64_t> == DType::Int64);
  static_assert(flashgrad::dtype_of<bool> == DType::Bool);
}

TEST(DType, DispatchAllSelectsMatchingType) {
  for (const DType dtype :
       {DType::Bool, DType::Int32, DType::Int64, DType::Float32, DType::Float64}) {
    const DType seen = flashgrad::dispatch_all(
        dtype, "test", [&]<typename T>() { return flashgrad::dtype_of<T>; });
    EXPECT_EQ(seen, dtype);
  }
}

TEST(DType, DispatchNumericRejectsBool) {
  EXPECT_THROW(flashgrad::dispatch_numeric(DType::Bool, "test", []<typename T>() {}),
               flashgrad::Error);
  EXPECT_NO_THROW(flashgrad::dispatch_numeric(DType::Int32, "test", []<typename T>() {}));
}

TEST(DType, DispatchFloatingRejectsIntegers) {
  EXPECT_THROW(flashgrad::dispatch_floating(DType::Int64, "test", []<typename T>() {}),
               flashgrad::Error);
  const bool is_double = flashgrad::dispatch_floating(
      DType::Float64, "test", []<typename T>() { return std::is_same_v<T, double>; });
  EXPECT_TRUE(is_double);
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
