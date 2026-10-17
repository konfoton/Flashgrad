// Tour of the Phase 2 ops: elementwise kernels, broadcasting, reductions and strided inputs,
// composed into a numerically stable softmax and a cross-entropy loss. Needs a GPU.
#include <iostream>

#include "flashgrad/flashgrad.h"

namespace fg = flashgrad;

// numerically stable softmax(x) along `dim`.
fg::Tensor softmax(const fg::Tensor& x, std::int64_t dim) {
  const fg::Tensor shifted = fg::add(x, fg::neg(fg::max(x, dim, /*keepdim=*/true)));
  const fg::Tensor e = fg::exp(shifted);
  return fg::mul(e, fg::reciprocal(fg::sum(e, dim, /*keepdim=*/true)));
}

int main() {
  if (!fg::cuda::is_available()) {
    std::cout << "no CUDA device available, skipping\n";
    return 0;
  }
  const fg::Device gpu = fg::Device::cuda(0);

  const fg::Tensor x = fg::arange(-3.0, 3.0).view({2, 3}).cuda();
  std::cout << "x =\n" << x << "\n";
  std::cout << "relu(x) =\n" << fg::relu(x) << "\n";
  std::cout << "sigmoid(x) =\n" << fg::sigmoid(x) << "\n";
  const fg::Tensor bias = fg::tensor({10.0f, 20.0f, 30.0f}, gpu);  // [3] broadcasts to [2, 3]
  std::cout << "x + bias =\n" << fg::add(x, bias) << "\n";
  std::cout << "x < 0 =\n" << fg::lt(x, fg::zeros({}, {.device = gpu})) << "\n";

  // Strided inputs: views go straight into the kernels, no copy first
  std::cout << "x.t() * x.t() =\n" << fg::mul(x.t(), x.t()) << "\n";

  std::cout << "sum(x, 0) = " << fg::sum(x, 0) << "\n";
  std::cout << "max(x, 1) = " << fg::max(x, 1) << "\n";
  std::cout << "prod(x, 1, keepdim) =\n" << fg::prod(x, 1, /*keepdim=*/true) << "\n";

  fg::manual_seed(0);
  const fg::Tensor logits = fg::mul(fg::randn({4, 5}).cuda(), fg::full({}, 3.0, {.device = gpu}));
  const fg::Tensor labels = fg::tensor({1.0f, 0.0f, 4.0f, 2.0f}, {4, 1}, gpu);
  const fg::Tensor classes = fg::arange(5.0).view({1, 5}).cuda();
  const fg::Tensor one_hot = fg::eq(labels, classes);  // [4, 1] vs [1, 5] -> [4, 5]

  const fg::Tensor probs = softmax(logits, 1);
  std::cout << "softmax(logits) =\n" << probs << "\n";
  std::cout << "row sums (all 1) = " << fg::sum(probs, 1) << "\n";

  // loss_i = -log p_i[label_i]; the one-hot mask picks out the label's probability.
  const fg::Tensor picked = fg::sum(fg::mul(probs, one_hot), 1);
  const fg::Tensor loss = fg::neg(fg::log(picked));
  std::cout << "cross-entropy per sample = " << loss << "\n";
  std::cout << "mean loss = " << fg::sum(loss, 0).item() / 4.0f << "\n";

  // A sample is correct when its label's logit equals the row max.
  const fg::Tensor is_max = fg::eq(logits, fg::max(logits, 1, /*keepdim=*/true));
  const float correct = fg::sum(fg::sum(fg::mul(is_max, one_hot), 1), 0).item();
  std::cout << "accuracy = " << correct << " / 4\n";
  return 0;
}
