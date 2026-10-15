#include <gtest/gtest.h>

#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/tensor.h"

namespace {

using flashgrad::Shape;
using flashgrad::Strides;
using flashgrad::Tensor;

// 0, 1, ..., n-1 as float32 with the given shape.
Tensor iota(const Shape& shape) {
  return flashgrad::arange(static_cast<double>(flashgrad::numel(shape))).view(shape);
}

std::vector<float> values(const Tensor& t) { return t.to_vector(); }

TEST(Views, ViewSharesStorage) {
  const Tensor a = iota({2, 6});
  const Tensor v = a.view({3, 4});
  EXPECT_TRUE(v.storage().is_same(a.storage()));
  EXPECT_EQ(v.strides(), (Strides{4, 1}));
  v.fill_(1.0);
  EXPECT_EQ(values(a), std::vector<float>(12, 1.0f));
}

TEST(Views, ViewInfersOneDimension) {
  const Tensor a = iota({2, 6});
  EXPECT_EQ(a.view({-1}).shape(), (Shape{12}));
  EXPECT_EQ(a.view({4, -1}).shape(), (Shape{4, 3}));
  EXPECT_EQ(a.view({-1, 1, 2}).shape(), (Shape{6, 1, 2}));
  EXPECT_THROW(a.view({-1, -1}), flashgrad::Error);
  EXPECT_THROW(a.view({5, -1}), flashgrad::Error);
  EXPECT_THROW(a.view({5}), flashgrad::Error);
}

TEST(Views, ViewOfScalarAndToScalar) {
  const Tensor s = flashgrad::full({}, 3.0);
  EXPECT_EQ(s.view({1, 1}).shape(), (Shape{1, 1}));
  EXPECT_EQ(s.view({1, 1}).view({}).item(), 3.0f);
}

TEST(Views, ViewOfTransposedThrows) {
  const Tensor t = iota({2, 3}).t();
  EXPECT_FALSE(t.is_contiguous());
  EXPECT_THROW(t.view({6}), flashgrad::Error);
}

TEST(Views, ViewOfCompatibleNonContiguousTensor) {
  // Every other row of a [4, 2, 3] tensor: rows are not adjacent, but each row's [2, 3]
  // block is contiguous, so those two dims can still be merged.
  const Tensor a = iota({4, 2, 3}).slice(0, 0, 4, 2);
  EXPECT_FALSE(a.is_contiguous());
  const Tensor v = a.view({2, 6});
  EXPECT_TRUE(v.storage().is_same(a.storage()));
  EXPECT_EQ(v.strides(), (Strides{12, 1}));
  EXPECT_EQ(values(v), values(a.contiguous().view({2, 6})));
  EXPECT_EQ(a.view({2, 2, 3, 1}).shape(), (Shape{2, 2, 3, 1}));
  EXPECT_THROW(a.view({12}), flashgrad::Error);
}

TEST(Views, ReshapeCopiesOnlyWhenNeeded) {
  const Tensor a = iota({2, 3});
  EXPECT_TRUE(a.reshape({3, 2}).storage().is_same(a.storage()));
  const Tensor r = a.t().reshape({6});
  EXPECT_FALSE(r.storage().is_same(a.storage()));
  EXPECT_EQ(values(r), (std::vector<float>{0, 3, 1, 4, 2, 5}));
}

TEST(Views, Flatten) {
  const Tensor a = iota({2, 3, 4});
  EXPECT_EQ(a.flatten().shape(), (Shape{24}));
  EXPECT_EQ(a.flatten(1).shape(), (Shape{2, 12}));
  EXPECT_EQ(a.flatten(0, 1).shape(), (Shape{6, 4}));
  EXPECT_EQ(a.flatten(-2, -1).shape(), (Shape{2, 12}));
  EXPECT_EQ(flashgrad::full({}, 1.0).flatten().shape(), (Shape{1}));
  EXPECT_THROW(a.flatten(2, 1), flashgrad::Error);
}

TEST(Views, PermuteAndTranspose) {
  const Tensor a = iota({2, 3, 4});
  const Tensor p = a.permute({2, 0, 1});
  EXPECT_EQ(p.shape(), (Shape{4, 2, 3}));
  EXPECT_EQ(p.strides(), (Strides{1, 12, 4}));
  EXPECT_TRUE(p.storage().is_same(a.storage()));
  // p[i][j][k] == a[j][k][i]
  EXPECT_EQ(p[3][1][2].item(), a[1][2][3].item());

  const Tensor t = a.transpose(0, -1);
  EXPECT_EQ(t.shape(), (Shape{4, 3, 2}));
  EXPECT_EQ(t[1][2][0].item(), a[0][2][1].item());

  EXPECT_THROW(a.permute({0, 1}), flashgrad::Error);
  EXPECT_THROW(a.permute({0, 1, 1}), flashgrad::Error);
  EXPECT_THROW(a.permute({0, 1, 3}), flashgrad::Error);
}

TEST(Views, T) {
  const Tensor m = iota({2, 3});
  EXPECT_EQ(m.t().shape(), (Shape{3, 2}));
  EXPECT_EQ(values(m.t()), (std::vector<float>{0, 3, 1, 4, 2, 5}));
  const Tensor v = iota({3});
  EXPECT_TRUE(v.t().is_same(v));
  EXPECT_THROW(iota({2, 2, 2}).t(), flashgrad::Error);
}

TEST(Views, UnsqueezeAndSqueeze) {
  const Tensor a = iota({2, 3});
  EXPECT_EQ(a.unsqueeze(0).shape(), (Shape{1, 2, 3}));
  EXPECT_EQ(a.unsqueeze(1).shape(), (Shape{2, 1, 3}));
  EXPECT_EQ(a.unsqueeze(2).shape(), (Shape{2, 3, 1}));
  EXPECT_EQ(a.unsqueeze(-1).shape(), (Shape{2, 3, 1}));
  EXPECT_TRUE(a.unsqueeze(1).is_contiguous());
  EXPECT_THROW(a.unsqueeze(3), flashgrad::Error);
  EXPECT_EQ(flashgrad::full({}, 1.0).unsqueeze(0).shape(), (Shape{1}));

  const Tensor b = iota({1, 2, 1, 3, 1});
  EXPECT_EQ(b.squeeze().shape(), (Shape{2, 3}));
  EXPECT_EQ(b.squeeze(0).shape(), (Shape{2, 1, 3, 1}));
  EXPECT_EQ(b.squeeze(-1).shape(), (Shape{1, 2, 1, 3}));
  EXPECT_EQ(b.squeeze(1).shape(), b.shape());  // size != 1: unchanged
  EXPECT_EQ(values(b.squeeze()), values(b));
}

TEST(Views, Expand) {
  const Tensor col = iota({3, 1});
  const Tensor e = col.expand({3, 4});
  EXPECT_EQ(e.shape(), (Shape{3, 4}));
  EXPECT_EQ(e.strides(), (Strides{1, 0}));
  EXPECT_EQ(values(e), (std::vector<float>{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}));

  const Tensor e2 = iota({3}).expand({2, -1});
  EXPECT_EQ(e2.shape(), (Shape{2, 3}));
  EXPECT_EQ(e2.strides(), (Strides{0, 1}));
  EXPECT_EQ(values(e2), (std::vector<float>{0, 1, 2, 0, 1, 2}));

  EXPECT_EQ(iota({1}).expand({0}).numel(), 0);
  EXPECT_EQ(flashgrad::full({}, 5.0).expand({2, 2}).to_vector(),
            (std::vector<float>(4, 5.0f)));
  EXPECT_THROW(iota({3}).expand({4}), flashgrad::Error);
  EXPECT_THROW(iota({2, 3}).expand({3}), flashgrad::Error);
  EXPECT_THROW(iota({3}).expand({-1, 3}), flashgrad::Error);
}

TEST(Views, Slice) {
  const Tensor a = iota({10});
  EXPECT_EQ(values(a.slice(0, 2, 5)), (std::vector<float>{2, 3, 4}));
  EXPECT_EQ(values(a.slice(0, 1, 10, 3)), (std::vector<float>{1, 4, 7}));
  EXPECT_EQ(values(a.slice(0, -3)), (std::vector<float>{7, 8, 9}));
  EXPECT_EQ(values(a.slice(0, 0, -8)), (std::vector<float>{0, 1}));
  EXPECT_EQ(values(a.slice(0, -100, 2)), (std::vector<float>{0, 1}));  // clamped
  EXPECT_EQ(a.slice(0, 5, 2).numel(), 0);
  EXPECT_EQ(a.slice(0, 10).numel(), 0);
  EXPECT_EQ(a.slice(0, 2, 5).offset(), 2);
  EXPECT_THROW(a.slice(0, 0, 5, 0), flashgrad::Error);
  EXPECT_THROW(a.slice(0, 0, 5, -1), flashgrad::Error);

  const Tensor m = iota({3, 4});
  const Tensor cols = m.slice(1, 1, 4, 2);  // columns 1 and 3
  EXPECT_EQ(cols.shape(), (Shape{3, 2}));
  EXPECT_EQ(cols.strides(), (Strides{4, 2}));
  EXPECT_EQ(values(cols), (std::vector<float>{1, 3, 5, 7, 9, 11}));
}

TEST(Views, SelectAndIndex) {
  const Tensor m = iota({3, 4});
  EXPECT_EQ(values(m.select(0, 1)), (std::vector<float>{4, 5, 6, 7}));
  EXPECT_EQ(values(m.select(1, -1)), (std::vector<float>{3, 7, 11}));
  EXPECT_EQ(m[2][3].item(), 11.0f);
  EXPECT_EQ(m[2][3].dim(), 0);
  EXPECT_THROW(m.select(0, 3), flashgrad::Error);
  EXPECT_THROW(m.select(0, -4), flashgrad::Error);
  EXPECT_THROW(flashgrad::full({}, 1.0)[0], flashgrad::Error);
}

TEST(Views, WritesThroughViewsAreVisible) {
  const Tensor m = flashgrad::zeros({3, 3});
  m.select(0, 1).fill_(1.0);
  m.t().select(0, 2).fill_(2.0);  // last column
  EXPECT_EQ(values(m), (std::vector<float>{0, 0, 2, 1, 1, 2, 0, 0, 2}));
}

TEST(Views, ChainedViewsCompose) {
  const Tensor a = iota({4, 5, 6});
  const Tensor v = a.permute({2, 0, 1}).slice(0, 1, 6, 2).select(2, 3).t();  // [4, 3]
  ASSERT_EQ(v.shape(), (Shape{4, 3}));
  for (std::int64_t i = 0; i < 4; ++i) {
    for (std::int64_t j = 0; j < 3; ++j) {
      // v[i][j] = a.permute(2,0,1)[1 + 2j][i][3] = a[i][3][1 + 2j]
      EXPECT_EQ(v[i][j].item(), static_cast<float>(i * 30 + 3 * 6 + 1 + 2 * j));
    }
  }
}

TEST(Views, IsContiguous) {
  EXPECT_TRUE(iota({2, 3}).is_contiguous());
  EXPECT_FALSE(iota({2, 3}).t().is_contiguous());
  EXPECT_TRUE(iota({2, 3}).slice(0, 1).is_contiguous());
  EXPECT_FALSE(iota({2, 3}).slice(1, 1).is_contiguous());
  EXPECT_TRUE(iota({1, 3}).t().is_contiguous());  // size-1 dims don't matter
  EXPECT_FALSE(iota({3, 1}).expand({3, 2}).is_contiguous());
  EXPECT_TRUE(iota({0, 3}).t().is_contiguous());
}

TEST(Views, AsStridedChecksBounds) {
  const Tensor a = iota({6});
  EXPECT_EQ(values(a.as_strided({2, 2}, {1, 2}, 1)), (std::vector<float>{1, 3, 2, 4}));
  EXPECT_THROW(a.as_strided({2, 3}, {3, 1}, 1), flashgrad::Error);  // needs 7 elements
  EXPECT_THROW(a.as_strided({2}, {-1}, 1), flashgrad::Error);
  EXPECT_THROW(a.as_strided({2}, {1, 1}, 0), flashgrad::Error);
  EXPECT_NO_THROW(a.as_strided({0}, {1}, 100));  // nothing to access
}

}  // namespace
