#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "flashgrad/core/tensor.h"

namespace flashgrad::nn {

// Base class of every layer and model, like torch.nn.Module. A module owns named parameters
// (leaf tensors that require grad) and named submodules, and computes its output in forward().
// Subclasses register what they own in their constructor and keep handles to it:
//
//   class Scale : public Module {
//    public:
//     explicit Scale(std::int64_t n, TensorOptions options = {})
//         : weight_(register_parameter("weight", ones({n}, options))) {}
//     Tensor forward(const Tensor& x) override { return x * weight_; }
//
//    private:
//     Tensor weight_;
//   };
//
//   auto model = std::make_shared<Scale>(4, TensorOptions{.device = Device::cuda()});
//   Tensor y = (*model)(x);
//
// Modules live behind std::shared_ptr and can't be copied: a copy would share its parameters
// with the original (tensors are handles), so it wouldn't be an independent model.
// Parameters stay on the device they were created on; pass the device when constructing.
class Module {
 public:
  Module() = default;
  virtual ~Module() = default;
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

  // The output of the module for the input `x`.
  virtual Tensor forward(const Tensor& x) = 0;
  // Calls forward(x), so a module can be applied like a function: (*model)(x).
  Tensor operator()(const Tensor& x) { return forward(x); }

  // This module's own parameters in registration order, then the parameters of each submodule
  // (recursively) in registration order. The results are handles to the module's tensors:
  // updating them in place updates the module.
  std::vector<Tensor> parameters() const;
  // The same parameters, in the same order, with their names. A submodule's parameters get
  // the submodule's name and a '.' as a prefix, e.g. "encoder.fc1.weight".
  std::vector<std::pair<std::string, Tensor>> named_parameters() const;

  // Clears the gradient of every parameter, recursively: grad() becomes undefined.
  void zero_grad() const;

  // Sets training mode (`on`) or evaluation mode (!`on`) on this module and every submodule.
  // Some layers, like Dropout, behave differently in the two modes. New modules are in
  // training mode.
  void train(bool on = true);
  void eval() { train(false); }
  bool is_training() const { return training_; }

 protected:
  // Registers `tensor` as the parameter `name`, marks it as requiring grad and returns it.
  // `tensor` must be a defined leaf. `name` must be non-empty, must not contain '.', and must
  // not be taken by another parameter or submodule of this module. Throws flashgrad::Error
  // otherwise.
  Tensor register_parameter(const std::string& name, const Tensor& tensor);
  // Registers `module` as the submodule `name`. `module` must not be null; `name` follows
  // the same rules as for parameters.
  void register_module(const std::string& name, const std::shared_ptr<Module>& module);

 private:
  bool is_name_taken(const std::string& name) const;

  std::vector<std::pair<std::string, Tensor>> parameters_;
  std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
  bool training_ = true;
};

}  // namespace flashgrad::nn
