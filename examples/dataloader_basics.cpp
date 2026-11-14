// Tour of the data API: a Dataset holds examples on the host, a DataLoader stacks them into
// batches (in background threads when num_workers > 0) and moves the batches to the device.
// The library has no built-in datasets: a program defines its own, like MnistDataset below.
//
//   dataloader_basics              synthetic data, then MNIST if the build downloaded it
//   dataloader_basics <mnist-dir>  MNIST from another directory (the four unpacked IDX files)
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "flashgrad/flashgrad.h"

namespace {

namespace fg = flashgrad;

class MnistDataset : public fg::data::TensorDataset {
 public:
  explicit MnistDataset(const std::filesystem::path& root, bool train = true)
      : TensorDataset(load(root / (train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte"),
                           16, {train ? 60000 : 10000, 1, 28, 28}, 255.0F),
                      load(root / (train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte"),
                           8, {train ? 60000 : 10000}, 1.0F)) {}

 private:
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

}  // namespace

int main(int argc, char** argv) {
  const fg::Device device = fg::cuda::is_available() ? fg::Device::cuda() : fg::Device::cpu();

  // 10 examples with 3 features each; the target of example i is i.
  fg::manual_seed(0);
  const auto dataset =
      std::make_shared<fg::data::TensorDataset>(fg::randn({10, 3}), fg::arange(10.0));

  const fg::data::DataLoader loader(
      dataset, {.batch_size = 4, .shuffle = true, .num_workers = 2, .device = device});
  std::cout << loader.num_batches() << " batches per epoch\n";
  for (int epoch = 0; epoch < 2; ++epoch) {
    std::cout << "epoch " << epoch << "\n";
    for (const auto& [x, y] : loader) {
      std::cout << "  x = " << x << std::endl << "  y = " << y << std::endl;
    }
  }

  const std::filesystem::path mnist_dir = argc > 1 ? argv[1] : FLASHGRAD_MNIST_DIR;
  if (argc > 1 || std::filesystem::exists(mnist_dir / "train-images-idx3-ubyte")) {
    const auto mnist = std::make_shared<MnistDataset>(mnist_dir);
    std::cout << "MNIST: " << mnist->size() << " examples, images "
              << fg::describe(mnist->data()) << "\n";
    for (const int num_workers : {0, 4}) {
      const fg::data::DataLoader mnist_loader(
          mnist,
          {.batch_size = 64, .shuffle = true, .num_workers = num_workers, .device = device});
      const auto start = std::chrono::steady_clock::now();
      std::int64_t examples = 0;
      for (const auto& [images, labels] : mnist_loader) examples += labels.size(0);
      const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;
      std::cout << "  num_workers = " << num_workers << ": " << examples << " examples in "
                << seconds.count() << " s\n";
    }
  }
  return 0;
}
