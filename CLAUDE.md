# CLAUDE.md

Flashgrad is a deep learning library written from scratch in **C++20 and CUDA**, developed as an
engineering thesis (inżynierka). The goal is a PyTorch-like library (tensors, autograd,
`nn` modules, optimizers) with hand-written GPU kernels, culminating in FlashAttention and
training a tiny GPT.

## Key decisions (do not change without asking)

- **C++ is the only user-facing API.** No Python front-end. Python bindings (nanobind) are an
  optional last phase. Python may be used only in `tools/` for scripts that check results against
  PyTorch, never as a library dependency.
- **CUDA is the primary backend.** A simple CPU backend exists as the *reference
  implementation* for tests and debugging — keep it naive and obviously correct, not fast.
- **Own kernels are the point of the thesis.** cuBLAS/cuDNN may be used only as optional
  reference/fallback backends and for benchmark comparisons.
- Target hardware: RTX 3090 (`sm_86`), CUDA 13.0, GCC 13, CMake 3.28.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tests/flashgrad_tests --gtest_filter='Suite.*'   # run a subset

# Debug: device debug info + sync after every kernel
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DFLASHGRAD_SYNC_KERNELS=ON
cmake --build build-debug -j
compute-sanitizer --tool memcheck ./build-debug/tests/flashgrad_tests
```

The build uses `-Werror` for host C++ code — fix warnings, don't suppress them.
After any change, build **and** run the tests before saying the work is done. For new or changed
kernels, also run `compute-sanitizer` on the relevant tests.

## Layout

```
include/flashgrad/   public headers (core/, cuda/, later nn/, optim/)
src/core/            device-independent implementation
src/cpu/             CPU reference kernels
src/cuda/            CUDA kernels (.cu) and runtime helpers
tests/               GoogleTest, mirrors src/ (tests/core, tests/cuda, ...)
examples/            small executables; double as integration tests and docs
benchmarks/          kernel benchmarks (use cuda::EventTimer, never std::chrono for GPU work)
```

New source files must be added to the `flashgrad` target in the root `CMakeLists.txt`;
new test files to `tests/CMakeLists.txt`.

## Roadmap

Phases are tracked in `README.md` (Roadmap section) — tick them off there when done.
0 infra ✔ → 1 Tensor core → 2 CUDA kernels → 3 caching allocator → 4 autograd → 5 `nn` →
6 optimizers + data (MNIST) → 7 Conv2d + fused kernels → 8 FlashAttention / mixed precision /
tiny GPT → 9 benchmarks → 10 Python bindings (optional).
Don't start a phase before the previous phase's milestone works and is tested.

## Conventions

- Namespace `flashgrad`; CUDA-specific code in `flashgrad::cuda`; internals in `detail`.
- Style: Google-based `.clang-format` (100 cols). `snake_case` functions and variables,
  `PascalCase` types, `kConstant` constants, trailing `_` for private members.
- Headers use `#pragma once` and include what they use.
- **Errors:** validate inputs with `FG_CHECK(cond, "op: message ", value, ...)` and include
  the op name, shapes, dtype and device in the message. Every CUDA runtime call is wrapped in
  `FG_CUDA_CHECK(...)`, and every kernel launch is followed by `FG_CUDA_KERNEL_CHECK()`.
  Destructors must not throw (call `cudaFree` etc. without the check macro there).
- **Resources:** RAII only, no raw owning pointers or manual `cudaFree` in normal code.
- **Kernels:** use grid-stride loops or explicit bounds checks; use 64-bit indices
  (`std::int64_t`) for element counts and offsets; mark non-aliasing pointers `__restrict__`.
- **Asynchrony:** library ops must not call `cudaDeviceSynchronize`; synchronize only when
  results go back to the host (`item()`, copy to CPU).
- **Design for later phases now:** tensors are strided views from the start (shape, strides,
  offset); ops dispatch on `(device, dtype)`; APIs should accept a stream even if only the
  default stream is used initially.

## Testing rules

- Every op gets tests that compare the CUDA result with the CPU reference on random inputs,
  including edge cases: empty tensors, size 1, sizes that aren't multiples of the block size,
  non-contiguous inputs and broadcasting.
- Use exact comparison only where it's guaranteed (single fp32 ops); otherwise use
  relative + absolute tolerance.
- Autograd ops (Phase 4+) also need a finite-difference gradient check (in `float64` on the CPU).
- GPU tests must call `GTEST_SKIP()` when `flashgrad::cuda::is_available()` is false.
