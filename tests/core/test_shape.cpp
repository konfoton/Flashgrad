#include "flashgrad/core/shape.h"

#include <gtest/gtest.h>

#include "flashgrad/core/error.h"

namespace {

using flashgrad::Shape;
using flashgrad::Strides;

TEST(Shape, Numel) {
  EXPECT_EQ(flashgrad::numel({}), 1);
  EXPECT_EQ(flashgrad::numel({5}), 5);
  EXPECT_EQ(flashgrad::numel({2, 3, 4}), 24);
  EXPECT_EQ(flashgrad::numel({2, 0, 4}), 0);
}

TEST(Shape, NumelRejectsNegativeAndOverflow) {
  EXPECT_THROW(flashgrad::numel({2, -1}), flashgrad::Error);
  EXPECT_THROW(flashgrad::numel({1LL << 40, 1LL << 40}), flashgrad::Error);
}

TEST(Shape, ContiguousStrides) {
  EXPECT_EQ(flashgrad::contiguous_strides({}), Strides{});
  EXPECT_EQ(flashgrad::contiguous_strides({7}), (Strides{1}));
  EXPECT_EQ(flashgrad::contiguous_strides({2, 3, 4}), (Strides{12, 4, 1}));
  // Empty dims don't zero out the strides of outer dims.
  EXPECT_EQ(flashgrad::contiguous_strides({2, 0, 3}), (Strides{3, 3, 1}));
}

TEST(Shape, Broadcast) {
  EXPECT_EQ(flashgrad::broadcast_shapes({2, 3}, {2, 3}), (Shape{2, 3}));
  EXPECT_EQ(flashgrad::broadcast_shapes({3, 1}, {1, 4}), (Shape{3, 4}));
  EXPECT_EQ(flashgrad::broadcast_shapes({5, 1, 4}, {3, 1}), (Shape{5, 3, 4}));
  EXPECT_EQ(flashgrad::broadcast_shapes({}, {2, 2}), (Shape{2, 2}));
  EXPECT_EQ(flashgrad::broadcast_shapes({0}, {1}), (Shape{0}));
  EXPECT_EQ(flashgrad::broadcast_shapes({2, 1}, {2, 0}), (Shape{2, 0}));
}

TEST(Shape, BroadcastErrorNamesOpAndShapes) {
  try {
    flashgrad::broadcast_shapes({2, 3}, {4}, "add");
    FAIL() << "expected an error";
  } catch (const flashgrad::Error& e) {
    const std::string msg = e.message();
    EXPECT_NE(msg.find("add"), std::string::npos) << msg;
    EXPECT_NE(msg.find("[2, 3]"), std::string::npos) << msg;
    EXPECT_NE(msg.find("[4]"), std::string::npos) << msg;
  }
}

TEST(Shape, NormalizeDim) {
  EXPECT_EQ(flashgrad::normalize_dim(0, 3, "op"), 0);
  EXPECT_EQ(flashgrad::normalize_dim(2, 3, "op"), 2);
  EXPECT_EQ(flashgrad::normalize_dim(-1, 3, "op"), 2);
  EXPECT_EQ(flashgrad::normalize_dim(-3, 3, "op"), 0);
  EXPECT_THROW(flashgrad::normalize_dim(3, 3, "op"), flashgrad::Error);
  EXPECT_THROW(flashgrad::normalize_dim(-4, 3, "op"), flashgrad::Error);
  EXPECT_THROW(flashgrad::normalize_dim(0, 0, "op"), flashgrad::Error);
}

TEST(Shape, ToString) {
  EXPECT_EQ(flashgrad::to_string({}), "[]");
  EXPECT_EQ(flashgrad::to_string({2, 3}), "[2, 3]");
}

}  // namespace
