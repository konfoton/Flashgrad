#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "flashgrad/autograd/grad_mode.h"
#include "flashgrad/autograd/node.h"
#include "flashgrad/core/error.h"
#include "flashgrad/core/ops.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/cuda/runtime.h"
#include "reference.h"

// The backward engine on whole graphs: shared inputs, branches, accumulation, freeing.
namespace {

#define SKIP_IF_NO_CUDA() \
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available"

namespace fg = flashgrad;
using fg::Tensor;

const fg::TensorOptions kGpu{.device = fg::Device::cuda()};

Tensor leaf(std::vector<float> values) {
  return fg::tensor(values, fg::Device::cuda()).requires_grad_();
}

TEST(Engine, RecordsOnlyWhatNeedsGradients) {
  SKIP_IF_NO_CUDA();
  const Tensor x = leaf({1, 2});
  const Tensor c = fg::ones({2}, kGpu);
  EXPECT_FALSE((c * c).requires_grad());
  const Tensor y = x * c;
  EXPECT_TRUE(y.requires_grad());
  EXPECT_FALSE(y.is_leaf());
  ASSERT_NE(y.grad_fn(), nullptr);
  EXPECT_EQ(y.grad_fn()->name(), "mul");
  ASSERT_EQ(y.grad_fn()->next().size(), 2u);
  EXPECT_NE(y.grad_fn()->next()[0].node, nullptr);  // x: its LeafGrad node
  EXPECT_EQ(y.grad_fn()->next()[0].node->name(), "LeafGrad");
  EXPECT_EQ(y.grad_fn()->next()[1].node, nullptr);  // c doesn't require grad
  {
    const fg::NoGradGuard no_grad;
    EXPECT_FALSE((x * c).requires_grad());
  }
  EXPECT_FALSE((x.detach() * c).requires_grad());
}

TEST(Engine, AnInputUsedTwiceGetsBothGradients) {
  SKIP_IF_NO_CUDA();
  const Tensor x = leaf({1, 2, 3});
  fg::sum(x * x).backward();  // d/dx x^2 = 2x
  EXPECT_EQ(x.grad().to_vector(), (std::vector<float>{2, 4, 6}));
}

TEST(Engine, DiamondGraph) {
  SKIP_IF_NO_CUDA();
  // y = exp(x) feeds both branches; its node must run once, after both sent their gradients.
  const Tensor x = leaf({0.5f, -1.0f});
  const Tensor y = fg::exp(x);
  const Tensor z = fg::sum(y * 3.0f + fg::sigmoid(y));
  z.backward();
  const std::vector<float> xv = x.to_vector();
  std::vector<float> expected;
  for (const float v : xv) {
    const double e = std::exp(double{v});
    const double s = 1.0 / (1.0 + std::exp(-e));
    expected.push_back(static_cast<float>((3.0 + s * (1.0 - s)) * e));
  }
  EXPECT_TRUE(ref::close(x.grad().to_vector(), expected, 1e-6, 1e-7));
}

TEST(Engine, BackwardFromANonScalarOutput) {
  SKIP_IF_NO_CUDA();
  const Tensor x = leaf({1, 2, 3});
  (x * 2.0f).backward(fg::tensor({1, 10, 100}, fg::Device::cuda()));
  EXPECT_EQ(x.grad().to_vector(), (std::vector<float>{2, 20, 200}));
}

TEST(Engine, GradientsAccumulateAcrossBackwardCalls) {
  SKIP_IF_NO_CUDA();
  const Tensor x = leaf({1, 2});
  fg::sum(x * 3.0f).backward();
  fg::sum(x * x).backward();
  EXPECT_EQ(x.grad().to_vector(), (std::vector<float>{5, 7}));  // 3 + 2x
  x.set_grad(Tensor());
  fg::sum(x).backward();
  EXPECT_EQ(x.grad().to_vector(), (std::vector<float>{1, 1}));
}

TEST(Engine, LeavesOutsideThePathGetNoGradient) {
  SKIP_IF_NO_CUDA();
  const Tensor x = leaf({1, 2});
  const Tensor unused = leaf({3, 4});
  const Tensor other = unused * 2.0f;  // a graph that doesn't lead to the loss
  fg::sum(x).backward();
  EXPECT_TRUE(x.grad().defined());
  EXPECT_FALSE(unused.grad().defined());
  EXPECT_FALSE(other.grad_fn()->released());
}

TEST(Engine, SeveralLeavesAndDetachedBranches) {
  SKIP_IF_NO_CUDA();
  const Tensor a = leaf({1, 2});
  const Tensor b = leaf({3, 4});
  // a * b, plus a branch that uses b only through detach(): no gradient to b from it.
  fg::sum(a * b + a.detach() * b.detach() * b.detach()).backward();
  EXPECT_EQ(a.grad().to_vector(), (std::vector<float>{3, 4}));
  EXPECT_EQ(b.grad().to_vector(), (std::vector<float>{1, 2}));
}

TEST(Engine, BackwardFreesTheGraphUnlessRetained) {
  SKIP_IF_NO_CUDA();
  const Tensor x = leaf({1, 2});
  const Tensor loss = fg::sum(x * x);
  loss.backward(Tensor(), /*retain_graph=*/true);
  EXPECT_FALSE(loss.grad_fn()->released());
  loss.backward();  // runs again, then frees
  EXPECT_EQ(x.grad().to_vector(), (std::vector<float>{4, 8}));
  EXPECT_TRUE(loss.grad_fn()->released());
  EXPECT_THROW(loss.backward(), fg::Error);
  // A second loss sharing part of the freed graph can't go through it either.
  const Tensor h = x * 2.0f;
  fg::sum(h).backward();
  EXPECT_THROW(fg::sum(h * h).backward(), fg::Error);
}

TEST(Engine, TheGraphDiesWithItsOutputs) {
  SKIP_IF_NO_CUDA();
  const Tensor x = leaf({1, 2});
  std::weak_ptr<fg::autograd::Node> node;
  {
    const Tensor y = fg::exp(x) * 2.0f;
    node = y.grad_fn();
    EXPECT_FALSE(node.expired());
  }
  EXPECT_TRUE(node.expired());
  // The LeafGrad node belongs to the leaf, not the graph.
  std::weak_ptr<fg::autograd::Node> leaf_node;
  {
    const Tensor y = x * 2.0f;
    leaf_node = y.grad_fn()->next()[0].node;
  }
  EXPECT_FALSE(leaf_node.expired());
}

TEST(Engine, LongChains) {
  SKIP_IF_NO_CUDA();
  // Backward is iterative, so a deep graph doesn't recurse.
  const Tensor x = leaf({1});
  Tensor y = x;
  for (int i = 0; i < 5000; ++i) y = y * 1.0f + 0.0f;
  fg::sum(y).backward();
  EXPECT_EQ(x.grad().to_vector(), (std::vector<float>{1}));
}

TEST(Engine, BackwardRunsWithoutGradMode) {
  SKIP_IF_NO_CUDA();
  const Tensor x = leaf({1, 2});
  fg::sum(x * x).backward();
  EXPECT_FALSE(x.grad().requires_grad());  // first-order only
  EXPECT_TRUE(fg::is_grad_enabled());       // restored afterwards
}

}  // namespace
