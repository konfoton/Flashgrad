#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "flashgrad/autograd/node.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/tensor.h"

namespace {

namespace fg = flashgrad;
using fg::Tensor;
using fg::autograd::Node;

TEST(TensorAutograd, PlainTensorsDoNotRequireGrad) {
  const Tensor t = fg::ones({2, 3});
  EXPECT_FALSE(t.requires_grad());
  EXPECT_TRUE(t.is_leaf());
  EXPECT_EQ(t.grad_fn(), nullptr);
  EXPECT_FALSE(t.grad().defined());
  EXPECT_EQ(t.impl()->autograd_meta(), nullptr);  // no autograd state allocated
}

TEST(TensorAutograd, RequiresGradOnLeaves) {
  const Tensor t = fg::ones({2, 3});
  EXPECT_TRUE(t.requires_grad_().is_same(t));
  EXPECT_TRUE(t.requires_grad());
  EXPECT_TRUE(t.is_leaf());
  t.requires_grad_(false);
  EXPECT_FALSE(t.requires_grad());
  // Handles share the TensorImpl, so they share the flag.
  const Tensor alias = t;
  alias.requires_grad_();
  EXPECT_TRUE(t.requires_grad());
}

TEST(TensorAutograd, SetGrad) {
  const Tensor t = fg::zeros({2, 3}).requires_grad_();
  const Tensor g = fg::ones({2, 3});
  t.set_grad(g);
  EXPECT_TRUE(t.grad().is_same(g));
  t.set_grad(Tensor());
  EXPECT_FALSE(t.grad().defined());
  EXPECT_THROW(t.set_grad(fg::ones({3, 2})), fg::Error);
  EXPECT_THROW(Tensor().set_grad(g), fg::Error);
}

TEST(TensorAutograd, DetachSharesDataWithoutHistory) {
  const Tensor t = fg::arange(6.0).view({2, 3}).t().requires_grad_();
  const Tensor d = t.detach();
  EXPECT_FALSE(d.requires_grad());
  EXPECT_TRUE(d.is_leaf());
  EXPECT_FALSE(d.is_same(t));
  EXPECT_TRUE(d.storage().is_same(t.storage()));
  EXPECT_EQ(d.shape(), t.shape());
  EXPECT_EQ(d.strides(), t.strides());
  EXPECT_EQ(d.offset(), t.offset());
  EXPECT_TRUE(t.requires_grad());  // the original is unchanged
}

TEST(TensorAutograd, UndefinedTensorsThrow) {
  EXPECT_THROW(Tensor().requires_grad(), fg::Error);
  EXPECT_THROW(Tensor().detach(), fg::Error);
}

TEST(Backward, OnALeafAccumulatesTheGradient) {
  const Tensor x = fg::zeros({3}).requires_grad_();
  const Tensor g = fg::tensor({1, 2, 3});
  x.backward(g);
  EXPECT_EQ(x.grad().to_vector(), (std::vector<float>{1, 2, 3}));
  EXPECT_FALSE(x.grad().storage().is_same(g.storage()));  // copied, not shared
  EXPECT_FALSE(x.grad().requires_grad());
}

TEST(Backward, ScalarOutputsDefaultToAGradientOfOne) {
  const Tensor x = fg::full({}, 5.0).requires_grad_();
  x.backward();
  EXPECT_EQ(x.grad().item(), 1.0f);
}

TEST(Backward, RejectsInvalidCalls) {
  EXPECT_THROW(fg::ones({2}).backward(fg::ones({2})), fg::Error);  // doesn't require grad
  const Tensor x = fg::ones({2}).requires_grad_();
  EXPECT_THROW(x.backward(), fg::Error);                // non-scalar without a gradient
  EXPECT_THROW(x.backward(fg::ones({3})), fg::Error);   // wrong shape
  EXPECT_THROW(Tensor().backward(), fg::Error);
}

TEST(AutogradNode, AppliesAndReleases) {
  int calls = 0;
  Node node("double", {fg::autograd::Edge{}}, [&](const Tensor& g) {
    ++calls;
    return std::vector<Tensor>{g};
  });
  EXPECT_EQ(node.name(), "double");
  EXPECT_EQ(node.next().size(), 1u);
  EXPECT_FALSE(node.released());
  const Tensor g = fg::ones({2});
  EXPECT_TRUE(node.apply(g)[0].is_same(g));
  node.release();
  EXPECT_TRUE(node.released());
  EXPECT_EQ(node.next().size(), 1u);  // edges stay
  EXPECT_THROW(node.apply(g), fg::Error);
  EXPECT_EQ(calls, 1);
}

TEST(AutogradNode, ChecksTheNumberOfGradients) {
  Node node("bad", {fg::autograd::Edge{}, fg::autograd::Edge{}},
            [](const Tensor& g) { return std::vector<Tensor>{g}; });
  EXPECT_THROW(node.apply(fg::ones({1})), fg::Error);
}

TEST(AutogradNode, PersistentNodesSurviveRelease) {
  Node node("AccumulateGrad", {}, [](const Tensor&) { return std::vector<Tensor>{}; },
            /*persistent=*/true);
  node.release();
  EXPECT_FALSE(node.released());
  EXPECT_NO_THROW(node.apply(fg::ones({1})));
}

}  // namespace
