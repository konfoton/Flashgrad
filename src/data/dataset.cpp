#include "flashgrad/data/dataset.h"

#include "flashgrad/core/error.h"

namespace flashgrad::data {

TensorDataset::TensorDataset(const Tensor& data, const Tensor& targets) {
  FG_CHECK(data.defined() && targets.defined(), "TensorDataset: got an undefined tensor");
  FG_CHECK(data.device().is_cpu() && targets.device().is_cpu(),
           "TensorDataset: tensors must be on the CPU (the DataLoader moves batches to the "
           "device), got ",
           describe(data), " and ", describe(targets));
  FG_CHECK(data.dim() >= 1 && targets.dim() >= 1,
           "TensorDataset: tensors need a leading example dimension, got ", describe(data),
           " and ", describe(targets));
  FG_CHECK(data.size(0) == targets.size(0), "TensorDataset: ", describe(data), " and ",
           describe(targets), " hold different numbers of examples");
  data_ = data.detach();
  targets_ = targets.detach();
}

std::int64_t TensorDataset::size() const { return data_.size(0); }

Example TensorDataset::get(std::int64_t index) const {
  FG_CHECK(index >= 0 && index < size(), "TensorDataset: index ", index,
           " out of range for a dataset of ", size(), " examples");
  return {data_.select(0, index), targets_.select(0, index)};
}

}  // namespace flashgrad::data
