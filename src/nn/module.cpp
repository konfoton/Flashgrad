#include "flashgrad/nn/module.h"

#include "flashgrad/core/error.h"

namespace flashgrad::nn {

Tensor Module::register_parameter(const std::string& name, const Tensor& tensor) {
  FG_CHECK(!name.empty(), "register_parameter: name is empty");
  FG_CHECK(name.find('.') == std::string::npos, "register_parameter: name '", name,
           "' contains '.'");
  FG_CHECK(!is_name_taken(name), "register_parameter: name '", name, "' is already taken");
  FG_CHECK(tensor.defined(), "register_parameter: tensor '", name, "' is undefined");
  FG_CHECK(tensor.is_leaf(), "register_parameter: ", describe(tensor), " is not a leaf");
  parameters_.emplace_back(name, tensor);
  return tensor.requires_grad_();
}

std::vector<std::pair<std::string, Tensor>> Module::named_parameters() const {
  std::vector<std::pair<std::string, Tensor>> out = parameters_;
  for (const auto& [child_name, child] : children_) {
    for (const auto& [parameter_name, parameter] : child->named_parameters()) {
      out.emplace_back(child_name + "." + parameter_name, parameter);
    }
  }
  return out;
}

std::vector<Tensor> Module::parameters() const {
  std::vector<Tensor> out;
  for (const auto& [parameter_name, parameter] : named_parameters()) {
    out.push_back(parameter);
  }
  return out;
}

void Module::register_module(const std::string& name, const std::shared_ptr<Module>& module) {
  FG_CHECK(!name.empty(), "register_module: name is empty");
  FG_CHECK(name.find('.') == std::string::npos, "register_module: name '", name,
           "' contains '.'");
  FG_CHECK(!is_name_taken(name), "register_module: name '", name, "' is already taken");
  FG_CHECK(module != nullptr, "register_module: module '", name, "' is null");
  children_.emplace_back(name, module);
}

void Module::zero_grad() const {
  for (const Tensor& parameter : parameters()) {
    parameter.set_grad(Tensor());
  }
}

void Module::train(bool on) {
  training_ = on;
  for (const auto& [child_name, child] : children_) {
    child->train(on);
  }
}

bool Module::is_name_taken(const std::string& name) const {
  for (const auto& [child_name, child] : children_) {
    if (name == child_name) return true;
  }
  for (const auto& [parameter_name, parameter] : parameters_) {
    if (name == parameter_name) return true;
  }
  return false;
}

}  // namespace flashgrad::nn
