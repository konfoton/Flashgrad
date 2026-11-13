#include "flashgrad/data/dataloader.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <map>
#include <mutex>
#include <numeric>
#include <span>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/shape.h"

namespace flashgrad::data {

namespace detail {

namespace {

// Stacks the examples at `indices` (at least one) into a CPU batch. Host copies only, so it
// may run in a worker thread. Returns early with an unfinished batch once `stop` is
// requested; the caller then throws the batch away.
Batch collate(const Dataset& dataset, std::span<const std::int64_t> indices,
              const std::stop_token& stop) {
  const std::int64_t count = static_cast<std::int64_t>(indices.size());

  // The example at `index`, validated and without autograd history (stacking is an
  // unrecorded copy).
  const auto get = [&dataset](std::int64_t index) -> Example {
    const Example example = dataset.get(index);
    FG_CHECK(example.data.defined() && example.target.defined(),
             "DataLoader: the dataset returned an undefined tensor for index ", index);
    FG_CHECK(example.data.device().is_cpu() && example.target.device().is_cpu(),
             "DataLoader: datasets must return CPU tensors, got ", describe(example.data),
             " and ", describe(example.target), " for index ", index);
    return {example.data.detach(), example.target.detach()};
  };
  // [count, ...example_shape]
  const auto batch_shape = [count](const Shape& example_shape) {
    Shape shape{count};
    shape.insert(shape.end(), example_shape.begin(), example_shape.end());
    return shape;
  };

  const Example first = get(indices[0]);
  Batch batch{empty(batch_shape(first.data.shape())), empty(batch_shape(first.target.shape()))};
  const auto put = [&batch](std::int64_t row, const Example& example) {
    batch.data.select(0, row).copy_(example.data);
    batch.target.select(0, row).copy_(example.target);
  };
  put(0, first);
  for (std::int64_t i = 1; i < count; ++i) {
    if (stop.stop_requested()) break;
    const std::int64_t index = indices[static_cast<std::size_t>(i)];
    const Example example = get(index);
    FG_CHECK(example.data.shape() == first.data.shape() &&
                 example.target.shape() == first.target.shape(),
             "DataLoader: examples of one batch have different shapes: index ", indices[0],
             " gave ", describe(first.data), " and ", describe(first.target), ", index ", index,
             " gave ", describe(example.data), " and ", describe(example.target));
    put(i, example);
  }
  return batch;
}

}  // namespace

// One pass over the dataset. Batch k holds the examples order[k * batch_size ...].
//
// With workers, the threads share two things and never talk to each other directly:
//  - `claimed_`, an atomic counter: a worker takes the next batch number from it, so every
//    batch is built by exactly one worker;
//  - `ready_`, a mutex-protected buffer of finished batches keyed by batch number, from which
//    the consumer takes them in order.
// A worker builds batch k only once k < next_ + window_ (next_ = the batch the consumer
// waits for), which bounds the number of finished batches held in memory. 
class Epoch {
 public:
  Epoch(std::shared_ptr<const Dataset> dataset, std::vector<std::int64_t> order,
        std::int64_t num_batches, const DataLoaderOptions& options)
      : dataset_(std::move(dataset)),
        order_(std::move(order)),
        batch_size_(options.batch_size),
        num_batches_(num_batches),
        window_(options.prefetch_batches > 0 ? options.prefetch_batches
                                             : 2 * std::int64_t{options.num_workers}),
        device_(options.device) {
    const std::int64_t num_workers = std::min<std::int64_t>(options.num_workers, num_batches_);
    workers_.reserve(static_cast<std::size_t>(num_workers));
    for (std::int64_t i = 0; i < num_workers; ++i) {
      workers_.emplace_back([this](const std::stop_token& stop) { work(stop); });
    }
  }

  ~Epoch() { finish(); }
  Epoch(const Epoch&) = delete;
  Epoch& operator=(const Epoch&) = delete;

  // Moves the next batch into `out`; false when the epoch is over. Rethrows the exception of
  // a batch that failed to build, after which the epoch is over.
  bool next(Batch& out) {
    if (done_) return false;
    if (next_ == num_batches_) {  // only the consumer writes next_, so it may read it unlocked
      finish();
      return false;
    }
    Slot slot;
    if (workers_.empty()) {
      try {
        slot.batch = collate(*dataset_, indices(next_), std::stop_token{});
      } catch (...) {
        finish();
        throw;
      }
      ++next_;
    } else {
      std::unique_lock lock(mutex_);
      ready_cv_.wait(lock, [this] { return ready_.contains(next_); });
      slot = std::move(ready_.extract(next_).mapped());
      ++next_;
      window_cv_.notify_all();  // the window moved: a waiting worker may start its batch
    }
    try {
      if (slot.error) std::rethrow_exception(slot.error);
      out.data = slot.batch.data.to(device_);
      out.target = slot.batch.target.to(device_);
    } catch (...) {
      finish();
      throw;
    }
    return true;
  }

 private:
  struct Slot {
    Batch batch;
    std::exception_ptr error;  // set instead of `batch` when building it threw
  };

  std::span<const std::int64_t> indices(std::int64_t batch) const {
    const std::int64_t begin = batch * batch_size_;
    const std::int64_t end =
        std::min(begin + batch_size_, static_cast<std::int64_t>(order_.size()));
    return std::span(order_).subspan(static_cast<std::size_t>(begin),
                                     static_cast<std::size_t>(end - begin));
  }

  void work(const std::stop_token& stop) {
    for (;;) {
      const std::int64_t batch = claimed_.fetch_add(1, std::memory_order_relaxed);
      if (batch >= num_batches_) return;

      {  // the mutex is held only while waiting for the window, not while building the batch
        std::unique_lock lock(mutex_);
        // Also woken by a stop request; false then.
        if (!window_cv_.wait(lock, stop, [&] { return batch < next_ + window_; })) return;
      }

      Slot slot;
      try {
        slot.batch = collate(*dataset_, indices(batch), stop);
      } catch (...) {
        slot.error = std::current_exception();
      }
      if (stop.stop_requested()) return;

      {
        const std::lock_guard lock(mutex_);
        ready_.emplace(batch, std::move(slot));
      }
      ready_cv_.notify_one();
    }
  }

  // Stops and joins the workers and drops the batches that were never consumed.
  void finish() noexcept {
    done_ = true;
    for (std::jthread& worker : workers_) worker.request_stop();
    workers_.clear();  // joins
    ready_.clear();
  }

  const std::shared_ptr<const Dataset> dataset_;
  const std::vector<std::int64_t> order_;
  const std::int64_t batch_size_;
  const std::int64_t num_batches_;
  const std::int64_t window_;
  const Device device_;

  bool done_ = false;                     // consumer only
  std::atomic<std::int64_t> claimed_{0};  // next batch a worker will build

  std::mutex mutex_;  // guards next_ and ready_
  std::int64_t next_ = 0;  // next batch the consumer takes
  std::map<std::int64_t, Slot> ready_;
  std::condition_variable ready_cv_;       // consumer: batch next_ arrived
  std::condition_variable_any window_cv_;  // workers: next_ advanced

  std::vector<std::jthread> workers_;  // last member: joined before the state above goes away
};

}  // namespace detail

DataLoader::DataLoader(std::shared_ptr<const Dataset> dataset, DataLoaderOptions options)
    : dataset_(std::move(dataset)), options_(options) {
  FG_CHECK(dataset_ != nullptr, "DataLoader: dataset is null");
  FG_CHECK(options_.batch_size > 0, "DataLoader: batch_size must be positive, got ",
           options_.batch_size);
  FG_CHECK(options_.num_workers >= 0, "DataLoader: num_workers must not be negative, got ",
           options_.num_workers);
  FG_CHECK(options_.prefetch_batches >= 0,
           "DataLoader: prefetch_batches must not be negative, got ", options_.prefetch_batches);
  const std::int64_t size = dataset_->size();
  FG_CHECK(size >= 0, "DataLoader: the dataset reports a negative size ", size);
  const bool partial_batch = !options_.drop_last && size % options_.batch_size != 0;
  num_batches_ = size / options_.batch_size + (partial_batch ? 1 : 0);
}

DataLoader::Iterator DataLoader::begin() const {
  if (num_batches_ == 0) return Iterator();
  // The order is drawn here, in the caller's thread: Generator isn't thread-safe, and this
  // makes the batches independent of the number of workers.
  std::vector<std::int64_t> order(static_cast<std::size_t>(dataset_->size()));
  std::iota(order.begin(), order.end(), std::int64_t{0});
  if (options_.shuffle) {
    Generator& generator =
        options_.generator != nullptr ? *options_.generator : default_generator();
    std::shuffle(order.begin(), order.end(), generator.engine());
  }
  return Iterator(
      std::make_shared<detail::Epoch>(dataset_, std::move(order), num_batches_, options_));
}

DataLoader::Iterator::Iterator(std::shared_ptr<detail::Epoch> epoch) : epoch_(std::move(epoch)) {
  advance();
}

const Batch& DataLoader::Iterator::operator*() const {
  FG_CHECK(epoch_ != nullptr, "DataLoader: dereferenced an iterator past the end of the epoch");
  return batch_;
}

const Batch* DataLoader::Iterator::operator->() const { return &**this; }

DataLoader::Iterator& DataLoader::Iterator::operator++() {
  FG_CHECK(epoch_ != nullptr, "DataLoader: advanced an iterator past the end of the epoch");
  advance();
  return *this;
}

void DataLoader::Iterator::operator++(int) { ++*this; }

bool operator==(const DataLoader::Iterator& it, std::default_sentinel_t) noexcept {
  return it.epoch_ == nullptr;
}

void DataLoader::Iterator::advance() {
  if (!epoch_->next(batch_)) {
    epoch_.reset();
    batch_ = Batch{};
  }
}

}  // namespace flashgrad::data
