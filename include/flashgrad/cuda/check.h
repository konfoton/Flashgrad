#pragma once

#include <cuda_runtime.h>

#include "flashgrad/core/error.h"

// FG_CUDA_CHECK(expr) — evaluates a CUDA runtime call and throws
// flashgrad::Error (with the CUDA error name and description) on failure.
#define FG_CUDA_CHECK(expr)                                                    \
  do {                                                                         \
    const cudaError_t fg_cuda_err_ = (expr);                                   \
    if (fg_cuda_err_ != cudaSuccess) [[unlikely]] {                            \
      ::flashgrad::detail::throw_error(                                        \
          __FILE__, __LINE__, nullptr,                                         \
          ::flashgrad::detail::concat("CUDA error ",                           \
                                      cudaGetErrorName(fg_cuda_err_), ": ",    \
                                      cudaGetErrorString(fg_cuda_err_),        \
                                      "\n  in: " #expr));                      \
    }                                                                          \
  } while (0)

// FG_CUDA_KERNEL_CHECK() — call right after a kernel launch.
// Always catches launch-configuration errors. With FLASHGRAD_SYNC_KERNELS
// it also synchronizes, so asynchronous faults (e.g. illegal memory access)
// are reported at the launch that caused them instead of much later.
#ifdef FLASHGRAD_SYNC_KERNELS
#define FG_CUDA_KERNEL_CHECK()                                                 \
  do {                                                                         \
    FG_CUDA_CHECK(cudaGetLastError());                                         \
    FG_CUDA_CHECK(cudaDeviceSynchronize());                                    \
  } while (0)
#else
#define FG_CUDA_KERNEL_CHECK() FG_CUDA_CHECK(cudaGetLastError())
#endif
