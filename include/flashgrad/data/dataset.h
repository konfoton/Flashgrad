#pragma once

#include <cstdint>

#include "flashgrad/core/tensor.h"

namespace flashgrad::data {

// One sample: the input and what the model should predict for it. Both are CPU tensors.
struct Example {
  Tensor data;
  Tensor target;
};

// A collection of examples addressed by index. Datasets hold host data only: a DataLoader
// reads them from worker threads, stacks the examples into batches and moves the batches to
// the device.
class Dataset {
 public:
  virtual ~Dataset() = default;

  virtual std::int64_t size() const = 0;

  // The example at `index`, in [0, size()). Both tensors must be on the CPU, and every index
  // must give the same shapes. Thread safe.
  virtual Example get(std::int64_t index) const = 0;
};

// Examples stored in two tensors: example i is (data[i], targets[i]), returned as views.
class TensorDataset : public Dataset {
 public:
  // CPU tensors with at least one dimension and the same size along dimension 0. The
  // dataset shares their storage (no copy) and drops their autograd history.
  TensorDataset(const Tensor& data, const Tensor& targets);

  std::int64_t size() const override;
  Example get(std::int64_t index) const override;

  const Tensor& data() const noexcept { return data_; }
  const Tensor& targets() const noexcept { return targets_; }

 private:
  Tensor data_;
  Tensor targets_;
};

}  // namespace flashgrad::data
