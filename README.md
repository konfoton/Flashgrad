# Flashgrad

A deep learning library written in C++ and CUDA: tensors, autograd, neural network
layers and optimizers, with hand-written GPU kernels.

## Requirements

- CUDA Toolkit 12.x or 13.x
- CMake ≥ 3.24
- GCC ≥ 11 (C++20)
- NVIDIA GPU, compute capability ≥ 7.0 (developed on RTX 3090, sm_86)

GoogleTest is downloaded automatically by CMake.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/examples/device_info
./build/examples/tensor_basics
./build/examples/ops_basics
./build/examples/autograd_basics
```

Debug build (device debug info + synchronize after every kernel launch, so CUDA
errors are reported at the kernel that caused them):

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DFLASHGRAD_SYNC_KERNELS=ON
cmake --build build-debug -j
```

### CMake options

| Option | Default | Description |
| --- | --- | --- |
| `CMAKE_CUDA_ARCHITECTURES` | `86` | Target GPU architecture(s) |
| `FLASHGRAD_BUILD_TESTS` | `ON` | Build GoogleTest suite |
| `FLASHGRAD_BUILD_EXAMPLES` | `ON` | Build example programs |
| `FLASHGRAD_SYNC_KERNELS` | `OFF` | `cudaDeviceSynchronize` after each kernel |
| `FLASHGRAD_WERROR` | `ON` | Treat host compiler warnings as errors |

## Usage

```cpp
#include "flashgrad/flashgrad.h"
namespace fg = flashgrad;

fg::Tensor a = fg::randn({64, 128});                  // float32, created on the CPU
fg::Tensor row = a[0];                                // views share storage
fg::Tensor cols = a.slice(1, 0, 128, 2);              // every other column, no copy
fg::Tensor d = a.t().contiguous().cuda();             // copy to the GPU
std::vector<float> host = d.to_vector();              // copies back to the host

fg::Tensor x = fg::randn({32, 10}).cuda();
fg::Tensor y = fg::sigmoid(x * 2.0f + fg::ones({10}).cuda());  // broadcasts [10]
fg::Tensor s = fg::sum(y, 1);                         // [32], reduced on the GPU

// Autograd: leaves that require grad get their gradients from backward().
fg::Tensor w = fg::randn({10, 3}, {.device = fg::Device::cuda()}).requires_grad_();
fg::Tensor loss = fg::mean(fg::relu(fg::matmul(x, w)));
loss.backward();                                      // w.grad() is [10, 3]
{
  fg::NoGradGuard no_grad;                            // updates aren't recorded
  w.sub_(w.grad() * 0.1f);
}
```

Tensors are strided views (shape, strides, offset) into reference-counted storage, like
PyTorch, and are all float32 for now. CPU tensors hold host data (creation, printing,
transfers); all math ops run as CUDA kernels: elementwise ops (neg, relu, exp, log, sigmoid,
reciprocal, add, sub, mul, div, maximum, lt, eq, with broadcasting and `+ - * /` operators,
scalars included), reductions over one dimension (sum, prod, max, mean) and over all elements
(sum, mean), log_softmax and a shared-memory tiled matmul for 2-D tensors; see
`include/flashgrad/core/ops.h`.

Every op and view except the comparisons and `as_strided` is differentiable. Ops on tensors
that require grad record a graph of nodes (`include/flashgrad/autograd/node.h`), and
`backward()` walks it in topological order, adding the gradients to the leaves' `grad()`.
Gradients are first order only, and in-place ops throw on tensors that require grad unless
grad mode is off (`NoGradGuard`).

## Debugging and profiling

GPU memory goes through a caching allocator: freed blocks are kept and reused instead of
being returned with `cudaFree`, which waits for the whole GPU. `flashgrad::cuda::memory_stats()`
reports what is allocated and cached, and `FLASHGRAD_NO_CACHING=1` turns caching off, which
compute-sanitizer needs to see out-of-bounds accesses inside cached blocks.

```bash
FLASHGRAD_NO_CACHING=1 compute-sanitizer --tool memcheck ./build-debug/tests/flashgrad_tests
nsys profile -o timeline ./build/examples/<program>
ncu --set full -o kernels ./build/examples/<program>
```

## Layout

```
include/flashgrad/   public headers
  autograd/          grad mode, graph nodes
  core/              error handling, dtype, device, storage, tensor, ops
  cuda/              CUDA runtime helpers, error checks, timers, allocator
src/
  autograd/          recording, backward engine
  core/              core implementation
  cpu/               host-side copy/fill for CPU tensors
  cuda/              CUDA kernels (.cu)
tests/               GoogleTest unit tests
examples/            small programs using the library
benchmarks/          kernel benchmarks
```

## Roadmap

- [x] Phase 0 — build system, error handling, CUDA helpers, tests
- [x] Phase 1 — Tensor core (storage, strides, views, host/device copies)
- [x] Phase 2 — CUDA kernels (elementwise, reductions, matmul, softmax)
- [x] Phase 3 — caching allocator
- [x] Phase 4 — autograd
- [ ] Phase 5 — `nn` modules
- [ ] Phase 6 — optimizers, data loading (MNIST)
- [ ] Phase 7 — Conv2d, fused kernels (CIFAR-10)
- [ ] Phase 8 — FlashAttention, mixed precision, tiny GPT
- [ ] Phase 9 — benchmarks
- [ ] Phase 10 — Python bindings (optional)
