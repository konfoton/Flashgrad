#include "flashgrad/core/random.h"

#include <cstdint>
#include <utility>

namespace flashgrad {

namespace {

template <typename Distribution>
Tensor sample(Shape shape, TensorOptions options, Generator* generator,
              Distribution distribution) {
  Generator& gen = generator != nullptr ? *generator : default_generator();
  Tensor out = empty(std::move(shape), {.dtype = options.dtype, .device = Device::cpu()});
  float* data = out.data_ptr();
  for (std::int64_t i = 0; i < out.numel(); ++i) {
    data[i] = static_cast<float>(distribution(gen.engine()));
  }
  return out.to(options.device);
}

}  // namespace

Generator& default_generator() {
  static Generator generator;
  return generator;
}

void manual_seed(std::uint64_t seed) { default_generator().manual_seed(seed); }

Tensor rand(Shape shape, TensorOptions options, Generator* generator) {
  return sample(std::move(shape), options, generator,
                std::uniform_real_distribution<double>(0.0, 1.0));
}

Tensor randn(Shape shape, TensorOptions options, Generator* generator) {
  return sample(std::move(shape), options, generator,
                std::normal_distribution<double>(0.0, 1.0));
}

}  // namespace flashgrad
