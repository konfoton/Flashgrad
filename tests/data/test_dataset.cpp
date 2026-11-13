#include "flashgrad/data/dataset.h"

#include <gtest/gtest.h>

#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/random.h"
#include "flashgrad/cuda/runtime.h"

namespace {

using flashgrad::Device;
using flashgrad::Shape;
using flashgrad::Tensor;
using flashgrad::data::Example;
using flashgrad::data::TensorDataset;

TEST(TensorDataset, ExamplesAreRows) {
  const Tensor data = flashgrad::arange(12.0).view({4, 3});
  const Tensor targets = flashgrad::tensor({10.0F, 11.0F, 12.0F, 13.0F});
  const TensorDataset dataset(data, targets);
  EXPECT_EQ(dataset.size(), 4);
  for (std::int64_t i = 0; i < 4; ++i) {
    const Example example = dataset.get(i);
    EXPECT_EQ(example.data.shape(), (Shape{3}));
    EXPECT_EQ(example.data.to_vector(), data[i].to_vector());
    EXPECT_EQ(example.target.dim(), 0);
    EXPECT_EQ(example.target.item(), 10.0F + static_cast<float>(i));
  }
}

TEST(TensorDataset, SharesStorage) {
  const Tensor data = flashgrad::zeros({2, 2});
  const Tensor targets = flashgrad::zeros({2});
  const TensorDataset dataset(data, targets);
  EXPECT_TRUE(dataset.get(1).data.storage().is_same(data.storage()));
  EXPECT_TRUE(dataset.get(1).target.storage().is_same(targets.storage()));
}

TEST(TensorDataset, NonContiguousData) {
  const Tensor data = flashgrad::arange(6.0).view({3, 2}).t();  // [2, 3], strides {1, 2}
  const TensorDataset dataset(data, flashgrad::zeros({2}));
  EXPECT_EQ(dataset.get(0).data.to_vector(), (std::vector<float>{0, 2, 4}));
  EXPECT_EQ(dataset.get(1).data.to_vector(), (std::vector<float>{1, 3, 5}));
}

TEST(TensorDataset, Empty) {
  const TensorDataset dataset(flashgrad::zeros({0, 3}), flashgrad::zeros({0}));
  EXPECT_EQ(dataset.size(), 0);
  EXPECT_THROW(dataset.get(0), flashgrad::Error);
}

TEST(TensorDataset, DropsAutogradHistory) {
  const Tensor data = flashgrad::zeros({2, 2}).requires_grad_();
  const TensorDataset dataset(data, flashgrad::zeros({2}));
  EXPECT_FALSE(dataset.data().requires_grad());
  EXPECT_FALSE(dataset.get(0).data.requires_grad());
}

TEST(TensorDataset, IndexOutOfRange) {
  const TensorDataset dataset(flashgrad::zeros({2, 2}), flashgrad::zeros({2}));
  EXPECT_THROW(dataset.get(2), flashgrad::Error);
  EXPECT_THROW(dataset.get(-1), flashgrad::Error);
}

TEST(TensorDataset, RejectsBadTensors) {
  EXPECT_THROW(TensorDataset(Tensor(), flashgrad::zeros({2})), flashgrad::Error);
  EXPECT_THROW(TensorDataset(flashgrad::zeros({2, 2}), Tensor()), flashgrad::Error);
  // Different numbers of examples.
  EXPECT_THROW(TensorDataset(flashgrad::zeros({2, 2}), flashgrad::zeros({3})), flashgrad::Error);
  // No example dimension.
  EXPECT_THROW(TensorDataset(flashgrad::zeros({}), flashgrad::zeros({})), flashgrad::Error);
}

TEST(TensorDataset, RejectsCudaTensors) {
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available";
  const Tensor on_device = flashgrad::zeros({2, 2}, {.device = Device::cuda()});
  EXPECT_THROW(TensorDataset(on_device, flashgrad::zeros({2})), flashgrad::Error);
  EXPECT_THROW(TensorDataset(flashgrad::zeros({2, 2}), on_device), flashgrad::Error);
}

}  // namespace
