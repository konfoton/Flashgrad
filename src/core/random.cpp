#include "flashgrad/core/random.h"

#include <cstdint>
#include <utility>

namespace flashgrad {

namespace {

template <typename Distribution>
Tensor sample(const char* op, Shape shape, TensorOptions options, Generator* generator,
              Distribution distribution) {
  FG_CHECK(is_floating(options.dtype), op, ": expected a floating-point dtype, got ",
           options.dtype);
  Generator& gen = generator != nullptr ? *generator : default_generator();
  Tensor out = empty(std::move(shape), {.dtype = options.dtype, .device = Device::cpu()});
  dispatch_floating(options.dtype, op, [&]<typename T>() {
    T* data = out.data_ptr<T>();
    for (std::int64_t i = 0; i < out.numel(); ++i) {
      data[i] = static_cast<T>(distribution(gen.engine()));
    }
  });
  return out.to(options.device);
}

}  // namespace

Generator& default_generator() {
  static Generator generator;
  return generator;
}

void manual_seed(std::uint64_t seed) { default_generator().manual_seed(seed); }

Tensor rand(Shape shape, TensorOptions options, Generator* generator) {
  return sample("rand", std::move(shape), options, generator,
                std::uniform_real_distribution<double>(0.0, 1.0));
}

Tensor randn(Shape shape, TensorOptions options, Generator* generator) {
  return sample("randn", std::move(shape), options, generator,
                std::normal_distribution<double>(0.0, 1.0));
}

}  // namespace flashgrad
