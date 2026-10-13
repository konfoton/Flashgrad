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

## Debugging and profiling

```bash
compute-sanitizer --tool memcheck ./build-debug/tests/flashgrad_tests
nsys profile -o timeline ./build/examples/<program>
ncu --set full -o kernels ./build/examples/<program>
```

## Layout

```
include/flashgrad/   public headers
  core/              error handling, (tensor, dtype, device — Phase 1)
  cuda/              CUDA runtime helpers, error checks, timers
src/
  core/              core implementation
  cpu/               CPU reference kernels
  cuda/              CUDA kernels (.cu)
tests/               GoogleTest unit tests
examples/            small programs using the library
benchmarks/          kernel benchmarks
```

## Roadmap

- [x] Phase 0 — build system, error handling, CUDA helpers, tests
- [ ] Phase 1 — Tensor core (storage, strides, views, CPU reference ops)
- [ ] Phase 2 — CUDA kernels (elementwise, reductions, matmul, softmax)
- [ ] Phase 3 — caching allocator
- [ ] Phase 4 — autograd
- [ ] Phase 5 — `nn` modules
- [ ] Phase 6 — optimizers, data loading (MNIST)
- [ ] Phase 7 — Conv2d, fused kernels (CIFAR-10)
- [ ] Phase 8 — FlashAttention, mixed precision, tiny GPT
- [ ] Phase 9 — benchmarks
- [ ] Phase 10 — Python bindings (optional)
