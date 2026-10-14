#pragma once

#include <cstdint>
#include <random>

#include "flashgrad/core/shape.h"
#include "flashgrad/core/tensor.h"

namespace flashgrad {

// Pseudo-random number source for tensor factories. Values are generated on the host, so
// the same seed gives the same tensor on every device. Not thread-safe.
class Generator {
 public:
  static constexpr std::uint64_t kDefaultSeed = 0x5eed;

  explicit Generator(std::uint64_t seed = kDefaultSeed) : engine_(seed) {}

  void manual_seed(std::uint64_t seed) { engine_.seed(seed); }
  std::mt19937_64& engine() noexcept { return engine_; }

 private:
  std::mt19937_64 engine_;
};

// The generator used when none is passed explicitly.
Generator& default_generator();
void manual_seed(std::uint64_t seed);  // reseeds default_generator()

// Uniform on [0, 1); floating-point dtypes only.
Tensor rand(Shape shape, TensorOptions options = {}, Generator* generator = nullptr);
// Standard normal; floating-point dtypes only.
Tensor randn(Shape shape, TensorOptions options = {}, Generator* generator = nullptr);

}  // namespace flashgrad
