#include "flashgrad/data/dataloader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "flashgrad/core/error.h"
#include "flashgrad/core/random.h"
#include "flashgrad/cuda/runtime.h"
#include "flashgrad/data/dataset.h"

namespace {

using flashgrad::Device;
using flashgrad::Generator;
using flashgrad::Shape;
using flashgrad::Tensor;
using flashgrad::data::Batch;
using flashgrad::data::DataLoader;
using flashgrad::data::DataLoaderOptions;
using flashgrad::data::Dataset;
using flashgrad::data::Example;
using flashgrad::data::TensorDataset;

static_assert(std::input_iterator<DataLoader::Iterator>);

// Example i is (data [i, i + 0.5], target i), so a batch tells which examples it holds.
// Options make it slow (by an amount that varies per index, so that workers finish their
// batches out of order), make one index throw, and count the calls.
class RangeDataset : public Dataset {
 public:
  explicit RangeDataset(std::int64_t size) : size_(size) {}

  std::int64_t size() const override { return size_; }

  Example get(std::int64_t index) const override {
    calls.fetch_add(1);
    if (jitter) std::this_thread::sleep_for(std::chrono::microseconds(50 * (index * 7919 % 7)));
    if (index == throw_at) throw std::runtime_error("RangeDataset: bad example");
    const float value = static_cast<float>(index);
    return {flashgrad::tensor({value, value + 0.5F}), flashgrad::tensor({value}).view({})};
  }

  bool jitter = false;
  std::int64_t throw_at = -1;
  mutable std::atomic<std::int64_t> calls{0};

 private:
  std::int64_t size_;
};

// The example indices of every batch of one epoch, after checking that data and target
// agree with each other.
std::vector<std::vector<std::int64_t>> epoch_indices(const DataLoader& loader) {
  std::vector<std::vector<std::int64_t>> batches;
  for (const auto& [data, target] : loader) {
    const std::int64_t count = target.size(0);
    EXPECT_EQ(data.shape(), (Shape{count, 2}));
    EXPECT_EQ(target.shape(), (Shape{count}));
    const std::vector<float> values = data.to_vector();
    const std::vector<float> targets = target.to_vector();
    std::vector<std::int64_t> indices;
    for (std::size_t i = 0; i < targets.size(); ++i) {
      EXPECT_EQ(values[2 * i], targets[i]);
      EXPECT_EQ(values[2 * i + 1], targets[i] + 0.5F);
      indices.push_back(static_cast<std::int64_t>(targets[i]));
    }
    batches.push_back(indices);
  }
  return batches;
}

std::vector<std::int64_t> flatten(const std::vector<std::vector<std::int64_t>>& batches) {
  std::vector<std::int64_t> all;
  for (const auto& batch : batches) all.insert(all.end(), batch.begin(), batch.end());
  return all;
}

std::vector<std::int64_t> iota(std::int64_t n) {
  std::vector<std::int64_t> v(static_cast<std::size_t>(n));
  std::iota(v.begin(), v.end(), std::int64_t{0});
  return v;
}

using Batches = std::vector<std::vector<std::int64_t>>;

// ---- Batching (in the calling thread and with workers) -------------------------------------

class DataLoaderWorkers : public testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(Workers, DataLoaderWorkers, testing::Values(0, 1, 2, 4, 8));

TEST_P(DataLoaderWorkers, SequentialBatches) {
  const DataLoader loader(std::make_shared<RangeDataset>(10),
                          {.batch_size = 4, .num_workers = GetParam()});
  EXPECT_EQ(loader.num_batches(), 3);
  EXPECT_EQ(epoch_indices(loader), (Batches{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9}}));
}

TEST_P(DataLoaderWorkers, DropLast) {
  const DataLoader loader(std::make_shared<RangeDataset>(10),
                          {.batch_size = 4, .drop_last = true, .num_workers = GetParam()});
  EXPECT_EQ(loader.num_batches(), 2);
  EXPECT_EQ(epoch_indices(loader), (Batches{{0, 1, 2, 3}, {4, 5, 6, 7}}));
}

TEST_P(DataLoaderWorkers, ExactMultipleOfBatchSize) {
  for (const bool drop_last : {false, true}) {
    const DataLoader loader(std::make_shared<RangeDataset>(6),
                            {.batch_size = 3, .drop_last = drop_last, .num_workers = GetParam()});
    EXPECT_EQ(epoch_indices(loader), (Batches{{0, 1, 2}, {3, 4, 5}}));
  }
}

TEST_P(DataLoaderWorkers, BatchLargerThanDataset) {
  const auto dataset = std::make_shared<RangeDataset>(3);
  const DataLoader loader(dataset, {.batch_size = 8, .num_workers = GetParam()});
  EXPECT_EQ(epoch_indices(loader), (Batches{{0, 1, 2}}));
  const DataLoader dropping(dataset,
                            {.batch_size = 8, .drop_last = true, .num_workers = GetParam()});
  EXPECT_EQ(dropping.num_batches(), 0);
  EXPECT_TRUE(dropping.begin() == dropping.end());
}

TEST_P(DataLoaderWorkers, SingleExample) {
  const DataLoader loader(std::make_shared<RangeDataset>(1),
                          {.batch_size = 1, .num_workers = GetParam()});
  EXPECT_EQ(epoch_indices(loader), (Batches{{0}}));
}

TEST_P(DataLoaderWorkers, EmptyDataset) {
  const DataLoader loader(std::make_shared<RangeDataset>(0),
                          {.batch_size = 4, .num_workers = GetParam()});
  EXPECT_EQ(loader.num_batches(), 0);
  EXPECT_TRUE(loader.begin() == loader.end());
  EXPECT_TRUE(epoch_indices(loader).empty());
}

TEST_P(DataLoaderWorkers, SlowDatasetKeepsOrder) {
  const auto dataset = std::make_shared<RangeDataset>(203);
  dataset->jitter = true;
  for (const int prefetch : {0, 1, 3}) {
    const DataLoader loader(
        dataset, {.batch_size = 5, .num_workers = GetParam(), .prefetch_batches = prefetch});
    EXPECT_EQ(flatten(epoch_indices(loader)), iota(203)) << "prefetch_batches = " << prefetch;
  }
}

TEST_P(DataLoaderWorkers, ShuffleMatchesSingleThread) {
  const auto dataset = std::make_shared<RangeDataset>(101);
  dataset->jitter = true;
  Generator reference_generator(42);
  const DataLoader reference(
      dataset, {.batch_size = 7, .shuffle = true, .generator = &reference_generator});
  Generator generator(42);
  const DataLoader loader(dataset, {.batch_size = 7,
                                    .shuffle = true,
                                    .num_workers = GetParam(),
                                    .generator = &generator});
  for (int epoch = 0; epoch < 3; ++epoch) {
    EXPECT_EQ(epoch_indices(loader), epoch_indices(reference)) << "epoch " << epoch;
  }
}

TEST_P(DataLoaderWorkers, DatasetExceptionReachesTheConsumer) {
  const auto dataset = std::make_shared<RangeDataset>(40);
  dataset->throw_at = 21;  // in batch 5 of 0..9
  const DataLoader loader(dataset, {.batch_size = 4, .num_workers = GetParam()});
  DataLoader::Iterator it = loader.begin();
  for (int batch = 0; batch < 4; ++batch) {
    EXPECT_EQ(it->target.to_vector().front(), static_cast<float>(4 * batch));
    ++it;
  }
  EXPECT_EQ(it->target.to_vector().front(), 16.0F);
  EXPECT_THROW(++it, std::runtime_error);
  // The epoch is over; the loader itself stays usable.
  ++it;
  EXPECT_TRUE(it == loader.end());
  EXPECT_THROW(*it, flashgrad::Error);
  EXPECT_THROW(++it, flashgrad::Error);
  dataset->throw_at = -1;
  EXPECT_EQ(flatten(epoch_indices(loader)), iota(40));
}

TEST_P(DataLoaderWorkers, ExceptionInFirstBatchIsThrownByBegin) {
  const auto dataset = std::make_shared<RangeDataset>(40);
  dataset->throw_at = 2;
  const DataLoader loader(dataset, {.batch_size = 4, .num_workers = GetParam()});
  EXPECT_THROW(loader.begin(), std::runtime_error);
}

TEST_P(DataLoaderWorkers, LeavingTheLoopEarlyStopsTheWorkers) {
  const auto dataset = std::make_shared<RangeDataset>(1000);
  dataset->jitter = true;
  const DataLoader loader(dataset, {.batch_size = 4, .num_workers = GetParam()});
  for (int repeat = 0; repeat < 30; ++repeat) {
    int seen = 0;
    for (const Batch& batch : loader) {
      EXPECT_EQ(batch.target.to_vector().front(), static_cast<float>(4 * seen));
      if (++seen == repeat % 4) break;  // 0: right after the first batch
    }
  }
  // An iterator that is never advanced.
  const DataLoader::Iterator unused = loader.begin();
  EXPECT_LT(dataset->calls.load(), 1000 * 31);
}

TEST_P(DataLoaderWorkers, TwoEpochsAtOnce) {
  const DataLoader loader(std::make_shared<RangeDataset>(12),
                          {.batch_size = 3, .num_workers = GetParam()});
  DataLoader::Iterator a = loader.begin();
  DataLoader::Iterator b = loader.begin();
  for (int batch = 0; batch < 4; ++batch) {
    EXPECT_EQ(a->target.to_vector(), b->target.to_vector());
    EXPECT_EQ(a->target.to_vector().front(), static_cast<float>(3 * batch));
    ++a;
    ++b;
  }
  EXPECT_TRUE(a == loader.end());
  EXPECT_TRUE(b == loader.end());
}

TEST_P(DataLoaderWorkers, MismatchedExampleShapes) {
  class Ragged : public Dataset {
   public:
    std::int64_t size() const override { return 8; }
    Example get(std::int64_t index) const override {
      return {flashgrad::zeros({index == 5 ? 3 : 2}), flashgrad::zeros({})};
    }
  };
  const DataLoader loader(std::make_shared<Ragged>(),
                          {.batch_size = 4, .num_workers = GetParam()});
  DataLoader::Iterator it = loader.begin();
  EXPECT_THROW(++it, flashgrad::Error);
}

TEST_P(DataLoaderWorkers, UndefinedExample) {
  class Undefined : public Dataset {
   public:
    std::int64_t size() const override { return 2; }
    Example get(std::int64_t) const override { return {flashgrad::zeros({2}), Tensor()}; }
  };
  const DataLoader loader(std::make_shared<Undefined>(), {.num_workers = GetParam()});
  EXPECT_THROW(loader.begin(), flashgrad::Error);
}

TEST_P(DataLoaderWorkers, TensorDatasetWithStridedAndEmptyExamples) {
  // Examples are columns of a matrix: non-contiguous views.
  const Tensor data = flashgrad::arange(15.0).view({3, 5}).t();  // [5, 3]
  const Tensor targets = flashgrad::zeros({5, 0});               // examples without elements
  const DataLoader loader(std::make_shared<TensorDataset>(data, targets),
                          {.batch_size = 2, .num_workers = GetParam()});
  std::vector<float> seen;
  std::int64_t count = 0;
  for (const auto& [x, y] : loader) {
    EXPECT_TRUE(x.is_contiguous());
    EXPECT_EQ(x.size(1), 3);
    EXPECT_EQ(y.shape(), (Shape{x.size(0), 0}));
    const std::vector<float> values = x.to_vector();
    seen.insert(seen.end(), values.begin(), values.end());
    count += x.size(0);
  }
  EXPECT_EQ(count, 5);
  EXPECT_EQ(seen, data.to_vector());
}

TEST_P(DataLoaderWorkers, MovesBatchesToCuda) {
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available";
  const auto dataset = std::make_shared<RangeDataset>(10);
  const DataLoader loader(
      dataset, {.batch_size = 4, .num_workers = GetParam(), .device = Device::cuda()});
  for (const auto& [data, target] : loader) {
    EXPECT_EQ(data.device(), Device::cuda());
    EXPECT_EQ(target.device(), Device::cuda());
  }
  EXPECT_EQ(epoch_indices(loader), (Batches{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9}}));
}

TEST_P(DataLoaderWorkers, RejectsCudaExamples) {
  if (!flashgrad::cuda::is_available()) GTEST_SKIP() << "no CUDA device available";
  class OnDevice : public Dataset {
   public:
    std::int64_t size() const override { return 2; }
    Example get(std::int64_t) const override { return {data_, flashgrad::zeros({})}; }

   private:
    Tensor data_ = flashgrad::zeros({2}, {.device = Device::cuda()});
  };
  const DataLoader loader(std::make_shared<OnDevice>(), {.num_workers = GetParam()});
  EXPECT_THROW(loader.begin(), flashgrad::Error);
}

// ---- Shuffling ----------------------------------------------------------------------------

TEST(DataLoader, ShuffleVisitsEveryExampleOnce) {
  Generator generator(1);
  const DataLoader loader(std::make_shared<RangeDataset>(50),
                          {.batch_size = 8, .shuffle = true, .generator = &generator});
  const std::vector<std::int64_t> first = flatten(epoch_indices(loader));
  const std::vector<std::int64_t> second = flatten(epoch_indices(loader));
  EXPECT_NE(first, iota(50));
  EXPECT_NE(first, second);  // a new order every epoch
  for (std::vector<std::int64_t> order : {first, second}) {
    std::sort(order.begin(), order.end());
    EXPECT_EQ(order, iota(50));
  }
}

TEST(DataLoader, ShuffleIsDeterministicPerSeed) {
  const auto dataset = std::make_shared<RangeDataset>(50);
  const auto epoch_with_seed = [&](std::uint64_t seed) {
    Generator generator(seed);
    return epoch_indices(
        DataLoader(dataset, {.batch_size = 8, .shuffle = true, .generator = &generator}));
  };
  EXPECT_EQ(epoch_with_seed(7), epoch_with_seed(7));
  EXPECT_NE(epoch_with_seed(7), epoch_with_seed(8));
}

TEST(DataLoader, ShuffleUsesDefaultGenerator) {
  const DataLoader loader(std::make_shared<RangeDataset>(50), {.batch_size = 8, .shuffle = true});
  flashgrad::manual_seed(3);
  const Batches first = epoch_indices(loader);
  flashgrad::manual_seed(3);
  EXPECT_EQ(epoch_indices(loader), first);
}

// ---- Prefetching ---------------------------------------------------------------------------

TEST(DataLoader, WorkersStayInsideThePrefetchWindow) {
  const auto dataset = std::make_shared<RangeDataset>(1000);
  const DataLoader loader(dataset,
                          {.batch_size = 4, .num_workers = 4, .prefetch_batches = 3});
  DataLoader::Iterator it = loader.begin();  // takes batch 0
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  // Batch 0 plus the three batches after it that the workers may build ahead.
  EXPECT_LE(dataset->calls.load(), 4 * (1 + 3));
  ++it;
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_LE(dataset->calls.load(), 4 * (2 + 3));
}

TEST(DataLoader, WorkersBuildBatchesAhead) {
  const auto dataset = std::make_shared<RangeDataset>(1000);
  const DataLoader loader(dataset, {.batch_size = 4, .num_workers = 2});  // window: 4 batches
  const DataLoader::Iterator it = loader.begin();
  for (int tries = 0; tries < 2000 && dataset->calls.load() < 4 * 5; ++tries) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(dataset->calls.load(), 4 * 5);  // without the consumer asking for more
}

// ---- Options -------------------------------------------------------------------------------

TEST(DataLoader, RejectsBadOptions) {
  const auto dataset = std::make_shared<RangeDataset>(4);
  EXPECT_THROW(DataLoader(nullptr), flashgrad::Error);
  EXPECT_THROW(DataLoader(dataset, {.batch_size = 0}), flashgrad::Error);
  EXPECT_THROW(DataLoader(dataset, {.batch_size = -2}), flashgrad::Error);
  EXPECT_THROW(DataLoader(dataset, {.num_workers = -1}), flashgrad::Error);
  EXPECT_THROW(DataLoader(dataset, {.prefetch_batches = -1}), flashgrad::Error);
}

TEST(DataLoader, DropsAutogradHistoryOfExamples) {
  class RequiresGrad : public Dataset {
   public:
    std::int64_t size() const override { return 3; }
    Example get(std::int64_t) const override {
      return {flashgrad::ones({2}).requires_grad_(), flashgrad::zeros({})};
    }
  };
  const DataLoader loader(std::make_shared<RequiresGrad>(), {.batch_size = 3});
  const DataLoader::Iterator it = loader.begin();
  EXPECT_FALSE(it->data.requires_grad());
  EXPECT_EQ(it->data.to_vector(), std::vector<float>(6, 1.0F));
}

}  // namespace
