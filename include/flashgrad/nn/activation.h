#pragma once

#include <cstdint>
#include <memory>

#include "flashgrad/core/tensor.h"
#include "flashgrad/nn/module.h"

// Activation functions as modules, so they can be layers of a Sequential. None of them has
// parameters; each forward applies the function of the same name from core/ops.h or
// nn/functional.h.
namespace flashgrad::nn {

class ReLU : public Module {
 public:
  Tensor forward(const Tensor& x) override;
};

class Sigmoid : public Module {
 public:
  Tensor forward(const Tensor& x) override;
};

class Tanh : public Module {
 public:
  Tensor forward(const Tensor& x) override;
};

// The tanh approximation, like functional::gelu.
class GELU : public Module {
 public:
  Tensor forward(const Tensor& x) override;
};

class SiLU : public Module {
 public:
  Tensor forward(const Tensor& x) override;
};

class LeakyReLU : public Module {
 public:
  explicit LeakyReLU(float negative_slope = 0.01f) : negative_slope_(negative_slope) {}
  Tensor forward(const Tensor& x) override;
  float negative_slope() const { return negative_slope_; }

 private:
  float negative_slope_;
};

// Softmax along `dim` (negative dims count from the end).
class Softmax : public Module {
 public:
  explicit Softmax(std::int64_t dim) : dim_(dim) {}
  Tensor forward(const Tensor& x) override;
  std::int64_t dim() const { return dim_; }

 private:
  std::int64_t dim_;
};

// The activations that can be chosen by name, e.g. for the hidden layers of an MLP.
enum class Activation { kReLU, kSigmoid, kTanh, kGELU, kSiLU };

// A new module computing `activation`.
std::shared_ptr<Module> make_activation(Activation activation);

}  // namespace flashgrad::nn
