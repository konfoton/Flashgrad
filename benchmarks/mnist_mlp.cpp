// Phase 6 milestone: an MLP trained on MNIST with everything the library has so far — a
// user-defined Dataset, the DataLoader (batches built by background threads and moved to the
// GPU), nn modules, cross-entropy and Adam. Exits with an error if the test accuracy is below
// 97%. Needs a GPU and the MNIST files, which the build downloads.
//
//   mnist_mlp              MNIST from where the build put it
//   mnist_mlp <mnist-dir>  MNIST from another directory (the four unpacked IDX files)
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "flashgrad/flashgrad.h"

namespace fg = flashgrad;
namespace nn = flashgrad::nn;
namespace optim = flashgrad::optim;

namespace {

// Everything there is to tune, in one place.
struct Config {
  std::int64_t hidden1 = 256;  // 784 -> hidden1 -> hidden2 -> 10
  std::int64_t hidden2 = 128;
  float dropout = 0.1F;
  int epochs = 5;
  std::int64_t batch_size = 128;
  std::int64_t eval_batch_size = 1000;
  float lr = 1e-3F;
  int num_workers = 4;
  int log_every = 100;  // steps between progress lines
  std::uint64_t seed = 0;
  float required_accuracy = 0.97F;
};

// MNIST from the four unpacked files in `root`. It fits in memory as two tensors, so it builds
// on TensorDataset, which already provides size() and get(): an example is an image
// [1, 28, 28] with pixels in [0, 1] and its class index as a 0-dim tensor.
class MnistDataset : public fg::data::TensorDataset {
 public:
  explicit MnistDataset(const std::filesystem::path& root, bool train = true)
      : TensorDataset(load(root / (train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte"),
                           16, {train ? 60000 : 10000, 1, 28, 28}, 255.0F),
                      load(root / (train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte"),
                           8, {train ? 60000 : 10000}, 1.0F)) {}

 private:
  // An MNIST file is a short header followed by one byte per pixel or label. Skips the
  // header and reads the bytes into a float tensor of `shape`, divided by `scale`.
  static fg::Tensor load(const std::filesystem::path& path, std::streamoff header_bytes,
                         fg::Shape shape, float scale) {
    const fg::Tensor out = fg::empty(std::move(shape));
    std::vector<unsigned char> bytes(static_cast<std::size_t>(out.numel()));
    std::ifstream file(path, std::ios::binary);
    file.seekg(header_bytes);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    FG_CHECK(file.gcount() == static_cast<std::streamsize>(bytes.size()),
             "MnistDataset: cannot read ", bytes.size(), " bytes from ", path,
             " (missing, or still gzip-compressed?)");
    float* data = out.data_ptr();
    for (std::size_t i = 0; i < bytes.size(); ++i) data[i] = static_cast<float>(bytes[i]) / scale;
    return out;
  }
};

// Number of rows whose largest logit is at the label's position.
float count_correct(const fg::Tensor& logits, const fg::Tensor& label) {
  const std::int64_t n = logits.size(0), classes = logits.size(1);
  const fg::Tensor one_hot =
      fg::eq(label.view({n, 1}), fg::arange(static_cast<double>(classes), logits.options()));
  const fg::Tensor is_max = fg::eq(logits, fg::max(logits, 1, /*keepdim=*/true));
  return fg::sum(is_max * one_hot).item();
}

// Accuracy of `model` over one pass of `loader`, in evaluation mode and without recording.
float evaluate(nn::Module& model, const fg::data::DataLoader& loader) {
  model.eval();
  const fg::NoGradGuard no_grad;
  float correct = 0.0F;
  for (const auto& [images, labels] : loader) {
    correct += count_correct(model(images.flatten(1)), labels);
  }
  return correct / static_cast<float>(loader.dataset().size());
}

}  // namespace

int main(int argc, char** argv) {
  const Config config;
  const std::filesystem::path mnist_dir = argc > 1 ? argv[1] : FLASHGRAD_MNIST_DIR;
  if (!fg::cuda::is_available()) {
    std::cout << "no CUDA device available, skipping\n";
    return 0;
  }
  if (argc == 1 && !std::filesystem::exists(mnist_dir / "train-images-idx3-ubyte")) {
    std::cout << "MNIST is not in " << mnist_dir << " (the build downloads it), skipping\n";
    return 0;
  }
  const fg::Device gpu = fg::Device::cuda();
  fg::manual_seed(config.seed);

  // Batches are built on the host by worker threads and arrive on the GPU.
  const fg::data::DataLoader train_loader(std::make_shared<MnistDataset>(mnist_dir, true),
                                          {.batch_size = config.batch_size,
                                           .shuffle = true,
                                           .num_workers = config.num_workers,
                                           .device = gpu});
  const fg::data::DataLoader test_loader(std::make_shared<MnistDataset>(mnist_dir, false),
                                         {.batch_size = config.eval_batch_size,
                                          .num_workers = config.num_workers,
                                          .device = gpu});

  nn::Sequential model{
      std::make_shared<nn::Linear>(784, config.hidden1, nn::LinearOptions{.device = gpu}),
      std::make_shared<nn::ReLU>(),
      std::make_shared<nn::Dropout>(config.dropout),
      std::make_shared<nn::Linear>(config.hidden1, config.hidden2,
                                   nn::LinearOptions{.device = gpu}),
      std::make_shared<nn::ReLU>(),
      std::make_shared<nn::Linear>(config.hidden2, 10, nn::LinearOptions{.device = gpu}),
  };
  optim::Adam optimizer(model.parameters(), {.lr = config.lr});
  std::cout << "training on " << train_loader.dataset().size() << " images, "
            << train_loader.num_batches() << " batches per epoch\n";

  int step = 0;
  for (int epoch = 1; epoch <= config.epochs; ++epoch) {
    model.train();
    for (const auto& [images, labels] : train_loader) {
      // [B, 1, 28, 28] -> [B, 784]
      const fg::Tensor loss = nn::cross_entropy(model(images.flatten(1)), labels);
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      // item() waits for the GPU, so the loss is only read now and then.
      if (++step % config.log_every == 0) {
        std::cout << "epoch " << epoch << ", step " << step << ": loss " << loss.item() << "\n";
      }
    }
    std::cout << "epoch " << epoch << ": test accuracy " << evaluate(model, test_loader) << "\n";
  }

  const float test_accuracy = evaluate(model, test_loader);
  if (test_accuracy < config.required_accuracy) {
    std::cerr << "training failed: test accuracy " << test_accuracy << " < "
              << config.required_accuracy << "\n";
    return 1;
  }
  return 0;
}
