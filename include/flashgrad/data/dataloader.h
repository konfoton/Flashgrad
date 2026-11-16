#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>

#include "flashgrad/core/device.h"
#include "flashgrad/core/random.h"
#include "flashgrad/core/tensor.h"
#include "flashgrad/data/dataset.h"

namespace flashgrad::data {

struct Batch {
  Tensor data;
  Tensor target;
};

struct DataLoaderOptions {
  std::int64_t batch_size = 1;
  bool shuffle = false;
  // Skip the last batch when it has fewer than batch_size examples.
  bool drop_last = false;
  // Background threads that build batches ahead of the training loop. 0 builds each batch
  // in the calling thread when it is asked for.
  int num_workers = 0;
  // How many batches the workers may build ahead of the one being consumed (this bounds the
  // host memory held by waiting batches). 0 means 2 * num_workers.
  int prefetch_batches = 0;
  Device device = Device::cpu();
  // Source of the shuffle order; null uses default_generator().
  Generator* generator = nullptr;
};

namespace detail {
class Epoch;
}  // namespace detail

// Iterates over a dataset in batches.
// Every begin() starts a new epoch. With num_workers > 0 the batches are built by background
// threads while the caller works on the previous ones; they still arrive in the same order,
// with the same contents, as with num_workers == 0. The workers only touch host memory: the
// copy to `device` happens in the thread that advances the iterator, so all CUDA work stays
// on one thread and one stream.
class DataLoader {
 public:
  class Iterator;  // defined below

  explicit DataLoader(std::shared_ptr<const Dataset> dataset, DataLoaderOptions options = {});

  const Dataset& dataset() const noexcept { return *dataset_; }
  const DataLoaderOptions& options() const noexcept { return options_; }
  std::int64_t num_batches() const noexcept { return num_batches_; }

  // Starts an epoch (drawing its order from the generator when shuffling) and returns an
  // iterator at its first batch. An iterator must be used by one thread only; several
  // epochs may be in progress at once.
  Iterator begin() const;
  std::default_sentinel_t end() const noexcept { return {}; }

 private:
  std::shared_ptr<const Dataset> dataset_;
  DataLoaderOptions options_;
  std::int64_t num_batches_ = 0;
};

// An input iterator over the batches of one epoch; compare with DataLoader::end(). Created by
// DataLoader::begin(), which starts the epoch; destroying the last copy stops its workers.
class DataLoader::Iterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = Batch;
  using difference_type = std::ptrdiff_t;
  using pointer = const Batch*;
  using reference = const Batch&;

  Iterator() = default;  // past the end

  const Batch& operator*() const;
  const Batch* operator->() const;
  // Fetches the next batch; rethrows an exception thrown by the dataset, which ends the epoch.
  Iterator& operator++();
  void operator++(int);

  friend bool operator==(const Iterator& it, std::default_sentinel_t) noexcept;

 private:
  friend class DataLoader;
  explicit Iterator(std::shared_ptr<detail::Epoch> epoch);
  void advance();

  std::shared_ptr<detail::Epoch> epoch_;  // null past the end
  Batch batch_;
};

}  // namespace flashgrad::data
