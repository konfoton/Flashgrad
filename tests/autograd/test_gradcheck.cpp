#include <gtest/gtest.h>

#include <vector>

#include "flashgrad/core/tensor.h"
#include "gradcheck.h"

// The checker itself, on the identity: backward of a leaf hands the output gradient through.
namespace {

namespace fg = flashgrad;
using fg::Tensor;

Tensor identity(const std::vector<Tensor>& in) { return in[0]; }

TEST(Gradcheck, AcceptsACorrectGradient) {
  const Tensor x = fg::tensor({1, -2, 3, 0.5f}, {2, 2});
  EXPECT_TRUE(gradcheck::check(
      identity, [](const std::vector<gradcheck::Values>& in) { return in[0]; }, {x}));
}

TEST(Gradcheck, RejectsAReferenceThatDisagrees) {
  const Tensor x = fg::tensor({1, -2, 3, 0.5f}, {2, 2});
  // The reference doubles its input: forward values and gradients both disagree.
  const auto doubled = [](const std::vector<gradcheck::Values>& in) {
    gradcheck::Values out = in[0];
    for (double& v : out) v *= 2.0;
    return out;
  };
  EXPECT_FALSE(gradcheck::check(identity, doubled, {x}));
}

TEST(Gradcheck, RejectsAReferenceOfTheWrongSize) {
  const Tensor x = fg::tensor({1, 2});
  const auto one_value = [](const std::vector<gradcheck::Values>&) {
    return gradcheck::Values{1.0};
  };
  EXPECT_FALSE(gradcheck::check(identity, one_value, {x}));
}

}  // namespace
