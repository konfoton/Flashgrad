// Phase 5 milestone: the task of autograd_basics.cpp (label = sign of x0 * x1, which a linear
// model can't separate) solved with the nn and optim modules instead of hand-written
// parameters. Trains a small network with Adam and checks it in evaluation mode on fresh
// points. Exits with an error if the test accuracy is below 90%. Needs a GPU.
#include <cstdint>
#include <iostream>
#include <memory>

#include "flashgrad/flashgrad.h"

namespace fg = flashgrad;
namespace nn = flashgrad::nn;
namespace optim = flashgrad::optim;

namespace {

struct Data {
  fg::Tensor x;      // [n, 2] points
  fg::Tensor label;  // [n] class index: 1 when x0 * x1 > 0, otherwise 0
};

Data make_data(std::int64_t n, const fg::TensorOptions& options) {
  const fg::Tensor x = fg::randn({n, 2}, options);
  const fg::Tensor label = fg::lt(fg::zeros({}, options), x.select(1, 0) * x.select(1, 1));
  return {x, label};
}

// Fraction of rows whose largest logit is at the label's position.
float accuracy(const fg::Tensor& logits, const fg::Tensor& label) {
  const std::int64_t n = logits.size(0), classes = logits.size(1);
  const fg::Tensor one_hot =
      fg::eq(label.view({n, 1}), fg::arange(static_cast<double>(classes), logits.options()));
  const fg::Tensor is_max = fg::eq(logits, fg::max(logits, 1, /*keepdim=*/true));
  return fg::sum(is_max * one_hot).item() / static_cast<float>(n);
}

}  // namespace

int main() {
  if (!fg::cuda::is_available()) {
    std::cout << "no CUDA device available, skipping\n";
    return 0;
  }
  const fg::Device gpu = fg::Device::cuda();
  fg::manual_seed(0);
  const Data train = make_data(512, {.device = gpu});
  const Data test = make_data(1024, {.device = gpu});

  // 2 -> 64 -> 64 -> 2, with dropout after the first hidden layer.
  nn::Sequential model{
      std::make_shared<nn::Linear>(2, 64, nn::LinearOptions{.device = gpu}),
      std::make_shared<nn::ReLU>(),
      std::make_shared<nn::Dropout>(0.1f),
      std::make_shared<nn::Linear>(64, 64, nn::LinearOptions{.device = gpu}),
      std::make_shared<nn::GELU>(),
      std::make_shared<nn::Linear>(64, 2, nn::LinearOptions{.device = gpu}),
  };
  std::cout << "model: " << model.parameters().size() << " parameter tensors\n";

  optim::Adam optimizer(model.parameters(), {.lr = 0.01f});
  constexpr int kSteps = 300;
  for (int step = 0; step <= kSteps; ++step) {
    model.train();
    const fg::Tensor logits = model(train.x);
    const fg::Tensor loss = nn::cross_entropy(logits, train.label);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    if (step % 50 == 0) {
      std::cout << "step " << step << ": loss " << loss.item() << ", train accuracy "
                << accuracy(logits.detach(), train.label) << "\n";
    }
  }

  // Evaluation: dropout off, and no graph is recorded.
  model.eval();
  const fg::NoGradGuard no_grad;
  const float test_accuracy = accuracy(model(test.x), test.label);
  std::cout << "test accuracy " << test_accuracy << "\n";
  if (test_accuracy < 0.9f) {
    std::cerr << "training failed: test accuracy " << test_accuracy << " < 0.9\n";
    return 1;
  }
  return 0;
}
