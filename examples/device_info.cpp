// Prints the visible CUDA devices and runs a small GPU sanity check.
#include <cstdlib>
#include <iostream>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/cuda/runtime.h"
#include "flashgrad/cuda/smoke.h"

int main() {
  namespace fgc = flashgrad::cuda;
  try {
    std::cout << "flashgrad " << FLASHGRAD_VERSION_STRING << "\n";

    const int count = fgc::device_count();
    if (count == 0) {
      std::cout << "No CUDA devices found.\n";
      return EXIT_FAILURE;
    }
    for (int i = 0; i < count; ++i) {
      std::cout << fgc::device_properties(i) << "\n";
    }

    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{10.0f, 20.0f, 30.0f};
    const auto c = fgc::smoke::vector_add(a, b);
    std::cout << "smoke test: [1,2,3] + [10,20,30] = [" << c[0] << "," << c[1] << "," << c[2]
              << "]\n";
    return EXIT_SUCCESS;
  } catch (const flashgrad::Error& e) {
    std::cerr << "flashgrad error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
