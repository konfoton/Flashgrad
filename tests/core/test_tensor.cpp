#include "flashgrad/core/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/random.h"

namespace {

using flashgrad::Device;
using flashgrad::DType;
using flashgrad::Shape;
using flashgrad::Strides;
using flashgrad::Tensor;

std::string str(const Tensor& t) {
  std::ostringstream oss;
  oss << t;
  return oss.str();
}

TEST(Tensor, DefaultIsUndefined) {
  const Tensor t;
  EXPECT_FALSE(t.defined());
  EXPECT_THROW(t.shape(), flashgrad::Error);
  EXPECT_THROW(t.numel(), flashgrad::Error);
  EXPECT_EQ(flashgrad::describe(t), "Tensor(undefined)");
  EXPECT_EQ(str(t), "tensor(undefined)");
}

TEST(Tensor, EmptyMetadata) {
  const Tensor t = flashgrad::empty({2, 3, 4});
  EXPECT_TRUE(t.defined());
  EXPECT_EQ(t.shape(), (Shape{2, 3, 4}));
  EXPECT_EQ(t.strides(), (Strides{12, 4, 1}));
  EXPECT_EQ(t.dim(), 3);
  EXPECT_EQ(t.size(1), 3);
  EXPECT_EQ(t.size(-1), 4);
  EXPECT_EQ(t.stride(0), 12);
  EXPECT_EQ(t.numel(), 24);
  EXPECT_EQ(t.offset(), 0);
  EXPECT_EQ(t.dtype(), DType::Float32);
  EXPECT_EQ(t.device(), Device::cpu());
  EXPECT_EQ(t.element_size(), 4u);
  EXPECT_EQ(t.nbytes(), 96u);
  EXPECT_EQ(t.storage().nbytes(), 96u);
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_THROW(t.size(3), flashgrad::Error);
}

TEST(Tensor, CpuMemoryIsAligned) {
  const Tensor t = flashgrad::empty({17});
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % flashgrad::kCpuAlignment, 0u);
}

TEST(Tensor, ScalarTensor) {
  const Tensor t = flashgrad::full({}, 2.5);
  EXPECT_EQ(t.dim(), 0);
  EXPECT_EQ(t.numel(), 1);
  EXPECT_TRUE(t.shape().empty());
  EXPECT_FLOAT_EQ(t.item(), 2.5f);
}

TEST(Tensor, ZeroElementTensor) {
  const Tensor t = flashgrad::zeros({3, 0});
  EXPECT_EQ(t.numel(), 0);
  EXPECT_EQ(t.data_ptr(), nullptr);
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_TRUE(t.to_vector().empty());
  EXPECT_EQ(t.clone().shape(), (Shape{3, 0}));
}

TEST(Tensor, CopiesShareData) {
  const Tensor a = flashgrad::zeros({3});
  const Tensor b = a;  // copies the handle, not the data
  b.fill_(7.0);
  EXPECT_EQ(a.to_vector(), (std::vector<float>{7, 7, 7}));
  EXPECT_TRUE(a.is_same(b));
}

TEST(Tensor, CloneIsDeep) {
  const Tensor a = flashgrad::ones({3});
  const Tensor b = a.clone();
  b.fill_(5.0);
  EXPECT_EQ(a.to_vector(), (std::vector<float>{1, 1, 1}));
  EXPECT_FALSE(a.storage().is_same(b.storage()));
}

TEST(Tensor, Factories) {
  EXPECT_EQ(flashgrad::zeros({2}).to_vector(), (std::vector<float>{0, 0}));
  EXPECT_EQ(flashgrad::ones({2}).to_vector(), (std::vector<float>{1, 1}));
  EXPECT_EQ(flashgrad::full({2}, 3.0).to_vector(), (std::vector<float>{3, 3}));

  const Tensor like = flashgrad::zeros({2, 2});
  EXPECT_EQ(flashgrad::ones_like(like).to_vector(), (std::vector<float>(4, 1)));
  EXPECT_EQ(flashgrad::full_like(like, 4.0).to_vector(), (std::vector<float>(4, 4)));
  EXPECT_EQ(flashgrad::empty_like(like).shape(), (Shape{2, 2}));
  EXPECT_EQ(flashgrad::zeros_like(flashgrad::ones({3})).to_vector(), (std::vector<float>(3, 0)));
}

TEST(Tensor, Arange) {
  EXPECT_EQ(flashgrad::arange(5.0).to_vector(), (std::vector<float>{0, 1, 2, 3, 4}));
  EXPECT_EQ(flashgrad::arange(1.0, 2.0, 0.25).to_vector(),
            (std::vector<float>{1.0f, 1.25f, 1.5f, 1.75f}));
  EXPECT_EQ(flashgrad::arange(5.0, 0.0, -2.0).to_vector(), (std::vector<float>{5, 3, 1}));
  EXPECT_EQ(flashgrad::arange(3.0, 3.0).numel(), 0);
  EXPECT_EQ(flashgrad::arange(3.0, 0.0).numel(), 0);  // empty, not negative
  EXPECT_THROW(flashgrad::arange(0.0, 1.0, 0.0), flashgrad::Error);
}

TEST(Tensor, Eye) {
  EXPECT_EQ(flashgrad::eye(3).to_vector(), (std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1}));
  EXPECT_EQ(flashgrad::eye(0).shape(), (Shape{0, 0}));
}

TEST(Tensor, FromValues) {
  const Tensor a = flashgrad::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  EXPECT_EQ(a.dtype(), DType::Float32);
  EXPECT_EQ(a.shape(), (Shape{2, 3}));
  EXPECT_EQ(a.to_vector(), (std::vector<float>{1, 2, 3, 4, 5, 6}));

  EXPECT_EQ(flashgrad::tensor({1, 2}).to_vector(), (std::vector<float>{1, 2}));  // int literals
  EXPECT_EQ(flashgrad::tensor(std::vector<float>{1, 2, 3}).shape(), (Shape{3}));
  EXPECT_EQ(flashgrad::tensor({7.0f}, Shape{}).item(), 7.0f);

  EXPECT_THROW(flashgrad::tensor({1.0f, 2.0f}, {3}), flashgrad::Error);
}

TEST(Tensor, ItemChecksSize) {
  EXPECT_EQ(flashgrad::tensor({3.75f}, {1, 1}).item(), 3.75f);
  EXPECT_THROW(flashgrad::zeros({2}).item(), flashgrad::Error);
}

TEST(Tensor, ToSameDeviceIsNoOp) {
  const Tensor f = flashgrad::tensor({-1.75f, 0.0f, 2.5f});
  EXPECT_TRUE(f.to(Device::cpu()).is_same(f));
  EXPECT_TRUE(f.cpu().is_same(f));
}

TEST(Tensor, FillRoundsValueToFloat) {
  const Tensor t = flashgrad::empty({3});
  t.fill_(0.1);
  EXPECT_EQ(t.to_vector(), (std::vector<float>(3, 0.1f)));
  t.zero_();
  EXPECT_EQ(t.to_vector(), (std::vector<float>{0, 0, 0}));
}

TEST(Tensor, FillNonContiguousView) {
  const Tensor t = flashgrad::zeros({3, 3});
  t.select(1, 1).fill_(1.0);  // middle column
  EXPECT_EQ(t.to_vector(), (std::vector<float>{0, 1, 0, 0, 1, 0, 0, 1, 0}));
}

TEST(Tensor, CopyBroadcasts) {
  const Tensor dst = flashgrad::zeros({2, 3});
  dst.copy_(flashgrad::tensor({1, 2, 3}));  // [3] -> [2, 3]
  EXPECT_EQ(dst.to_vector(), (std::vector<float>{1, 2, 3, 1, 2, 3}));
  EXPECT_THROW(dst.copy_(flashgrad::zeros({2})), flashgrad::Error);
  EXPECT_THROW(dst.copy_(flashgrad::zeros({3, 3})), flashgrad::Error);
}

TEST(Tensor, CopyIntoNonContiguousView) {
  const Tensor dst = flashgrad::zeros({2, 2});
  dst.t().copy_(flashgrad::tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}));
  EXPECT_EQ(dst.to_vector(), (std::vector<float>{1, 3, 2, 4}));
}

TEST(Tensor, CopyIntoExpandedViewThrows) {
  const Tensor base = flashgrad::zeros({1, 3});
  EXPECT_THROW(base.expand({2, 3}).copy_(flashgrad::ones({2, 3})), flashgrad::Error);
}

TEST(Tensor, CopyBetweenOverlappingViews) {
  // Shift right by one: a naive in-order loop would smear a[0] over everything.
  const Tensor a = flashgrad::arange(5.0);
  a.slice(0, 1).copy_(a.slice(0, 0, 4));
  EXPECT_EQ(a.to_vector(), (std::vector<float>{0, 0, 1, 2, 3}));
}

TEST(Tensor, CopyToItself) {
  const Tensor a = flashgrad::arange(4.0);
  a.copy_(a);
  EXPECT_EQ(a.to_vector(), (std::vector<float>{0, 1, 2, 3}));
}

TEST(Tensor, ContiguousReturnsSelfWhenPossible) {
  const Tensor a = flashgrad::zeros({2, 3});
  EXPECT_TRUE(a.contiguous().is_same(a));
  const Tensor t = a.t().contiguous();
  EXPECT_FALSE(t.storage().is_same(a.storage()));
  EXPECT_TRUE(t.is_contiguous());
}

TEST(Tensor, RandIsSeededAndInRange) {
  flashgrad::Generator g1(42);
  flashgrad::Generator g2(42);
  const Tensor a = flashgrad::rand({1000}, {}, &g1);
  const Tensor b = flashgrad::rand({1000}, {}, &g2);
  EXPECT_TRUE(flashgrad::equal(a, b));
  for (const float v : a.to_vector()) {
    EXPECT_GE(v, 0.0f);
    EXPECT_LT(v, 1.0f);
  }
}

TEST(Tensor, ManualSeedMakesDefaultGeneratorReproducible) {
  flashgrad::manual_seed(7);
  const Tensor a = flashgrad::randn({16});
  flashgrad::manual_seed(7);
  const Tensor b = flashgrad::randn({16});
  EXPECT_TRUE(flashgrad::equal(a, b));
}

TEST(Tensor, RandnStatistics) {
  flashgrad::Generator gen(1);
  const std::vector<float> v = flashgrad::randn({100000}, {}, &gen).to_vector();
  double mean = 0.0;
  for (const float x : v) mean += x;
  mean /= static_cast<double>(v.size());
  double var = 0.0;
  for (const float x : v) var += (x - mean) * (x - mean);
  var /= static_cast<double>(v.size());
  EXPECT_NEAR(mean, 0.0, 0.02);
  EXPECT_NEAR(var, 1.0, 0.02);
}

TEST(Tensor, Describe) {
  EXPECT_EQ(flashgrad::describe(flashgrad::zeros({2, 3})),
            "Tensor(shape=[2, 3], dtype=float32, device=cpu)");
}

TEST(Tensor, Printing) {
  EXPECT_EQ(str(flashgrad::tensor({1, 2, 3, 4}, {2, 2})),
            "tensor([[1, 2],\n        [3, 4]], dtype=float32, device=cpu)");
  EXPECT_EQ(str(flashgrad::full({}, 1.5)), "tensor(1.5, dtype=float32, device=cpu)");
  EXPECT_EQ(str(flashgrad::zeros({0, 2})), "tensor([], shape=[0, 2], dtype=float32, device=cpu)");
  // Non-contiguous views print in logical order.
  EXPECT_EQ(str(flashgrad::tensor({1, 2, 3, 4}, {2, 2}).t()),
            "tensor([[1, 3],\n        [2, 4]], dtype=float32, device=cpu)");
}

TEST(Tensor, PrintingSummarizesLargeTensors) {
  const std::string s = str(flashgrad::arange(2000.0));
  EXPECT_EQ(s, "tensor([0, 1, 2, ..., 1997, 1998, 1999], dtype=float32, device=cpu)");
}

TEST(Storage, SharedByViewsAndReleased) {
  flashgrad::Storage storage;
  {
    const Tensor a = flashgrad::zeros({4});
    const Tensor v = a.view({2, 2});
    EXPECT_TRUE(v.storage().is_same(a.storage()));
    storage = a.storage();
    EXPECT_EQ(storage.use_count(), 3);  // a, v and `storage`
  }
  EXPECT_EQ(storage.use_count(), 1);
  EXPECT_EQ(storage.nbytes(), 16u);
  EXPECT_EQ(storage.device(), Device::cpu());
}

}  // namespace
