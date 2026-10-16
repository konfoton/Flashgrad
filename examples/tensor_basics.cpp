// Tour of the Phase 1 tensor API: factories, views (no copies), copies and (if a GPU is
// present) moving data to the device and back. Math ops arrive as CUDA kernels in Phase 2.
#include <iostream>

#include "flashgrad/flashgrad.h"

int main() {
  namespace fg = flashgrad;

  const fg::Tensor a = fg::arange(6.0).view({2, 3});
  std::cout << "a =\n" << a << "\n";
  std::cout << "a.t() (a view, no copy) =\n" << a.t() << "\n";
  std::cout << "a[1] = " << a[1] << "\n";
  std::cout << "a.slice(1, 0, 3, 2) (every other column) =\n" << a.slice(1, 0, 3, 2) << "\n";
  std::cout << "row expanded to [3, 3] (stride 0) =\n" << a[0].expand({3, 3}) << "\n";
  std::cout << "a.t().reshape({6}) (copies: not viewable) = " << a.t().reshape({6}) << "\n";

  fg::manual_seed(0);
  std::cout << "randn({2, 2}) =\n" << fg::randn({2, 2}) << "\n";

  if (fg::cuda::is_available()) {
    const fg::Tensor d = a.cuda();
    std::cout << "on the GPU: " << d << "\n";
    std::cout << "transposed view downloaded: " << d.t().cpu() << "\n";
  }
  return 0;
}
