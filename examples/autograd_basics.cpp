// Phase 4 milestone: a two-layer network trained with autograd and plain SGD. The points
// are labeled by the sign of x0 * x1 (an XOR of the quadrants), which a linear model can't
// separate. Exits with an error if training doesn't reach 90% accuracy. Needs a GPU.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "flashgrad/flashgrad.h"

namespace fg = flashgrad;

int main() {
  if (!fg::cuda::is_available()) {
    std::cout << "no CUDA device available, skipping\n";
    return 0;
  }
  const fg::TensorOptions gpu{.device = fg::Device::cuda()};
  fg::manual_seed(0);

  constexpr std::int64_t n = 512, hidden = 32, classes = 2, steps = 600;
  constexpr float lr = 0.5f;
  const fg::Tensor x = fg::randn({n, 2}, gpu);
  const fg::Tensor label = fg::lt(fg::zeros({}, gpu), x.select(1, 0) * x.select(1, 1));
  const fg::Tensor one_hot = fg::eq(label.view({n, 1}), fg::arange(2.0, gpu).view({1, classes}));

  // Parameters are leaves: scaled random values, then marked as requiring grad.
  const fg::Tensor w1 = (fg::randn({2, hidden}, gpu) * 1.0f).requires_grad_();
  const fg::Tensor b1 = fg::zeros({hidden}, gpu).requires_grad_();
  const fg::Tensor w2 =
      (fg::randn({hidden, classes}, gpu) * std::sqrt(2.0f / hidden)).requires_grad_();
  const fg::Tensor b2 = fg::zeros({classes}, gpu).requires_grad_();
  const std::vector<fg::Tensor> params = {w1, b1, w2, b2};

  float accuracy = 0.0f;
  for (std::int64_t step = 0; step <= steps; ++step) {
    const fg::Tensor h = fg::relu(fg::matmul(x, w1) + b1);
    const fg::Tensor logits = fg::matmul(h, w2) + b2;
    const fg::Tensor loss = fg::mean(-fg::sum(fg::log_softmax(logits, 1) * one_hot, 1));

    loss.backward();  // fills p.grad() for every parameter
    {
      const fg::NoGradGuard no_grad;  // the update itself isn't part of the graph
      for (const fg::Tensor& p : params) {
        p.sub_(p.grad() * lr);
        p.set_grad(fg::Tensor());
      }
    }

    if (step % 100 == 0) {
      // A sample is right when its label's logit is the row max.
      const fg::Tensor out = logits.detach();
      const fg::Tensor is_max = fg::eq(out, fg::max(out, 1, /*keepdim=*/true));
      accuracy = fg::sum(is_max * one_hot).item() / static_cast<float>(n);
      std::cout << "step " << step << ": loss " << loss.item() << ", accuracy " << accuracy
                << "\n";
    }
  }
  if (accuracy < 0.9f) {
    std::cerr << "training failed: accuracy " << accuracy << " < 0.9\n";
    return 1;
  }
  return 0;
}
