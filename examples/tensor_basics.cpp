// Tour of the Phase 1 tensor API: factories, views, CPU ops and (if a GPU is present)
// moving data to the device and back.
#include <iostream>

#include "flashgrad/flashgrad.h"

int main() {
  namespace fg = flashgrad;

  const fg::Tensor a = fg::arange(6.0).view({2, 3});
  std::cout << "a =\n" << a << "\n";
  std::cout << "a.t() (a view, no copy) =\n" << a.t() << "\n";
  std::cout << "a[1] = " << a[1] << "\n";

  const fg::Tensor b = fg::ones({3});
  std::cout << "a + b (broadcast) =\n" << a + b << "\n";
  std::cout << "a @ a.t() =\n" << fg::matmul(a, a.t()) << "\n";
  std::cout << "sum(a, 1) = " << fg::sum(a, 1) << "\n";
  std::cout << "softmax(a, 1) =\n" << fg::softmax(a, 1) << "\n";

  fg::manual_seed(0);
  std::cout << "randn({2, 2}) =\n" << fg::randn({2, 2}) << "\n";

  if (fg::cuda::is_available()) {
    const fg::Tensor d = a.cuda();
    std::cout << "on the GPU: " << d << "\n";
    std::cout << "transposed view downloaded: " << d.t().cpu() << "\n";
  }
  return 0;
}
