#pragma once

#include <cuda_runtime.h>

#include "flashgrad/cuda/check.h"

namespace flashgrad::cuda {

// Measures GPU time between start() and stop() on a stream using CUDA events.
// Use this (not std::chrono) for kernel timing: launches are asynchronous.
//
//   EventTimer t;
//   t.start();
//   my_kernel<<<...>>>(...);
//   t.stop();
//   float ms = t.elapsed_ms();  // blocks until the stop event completes
class EventTimer {
 public:
  explicit EventTimer(cudaStream_t stream = nullptr) : stream_(stream) {
    FG_CUDA_CHECK(cudaEventCreate(&start_));
    FG_CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~EventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  EventTimer(const EventTimer&) = delete;
  EventTimer& operator=(const EventTimer&) = delete;

  void start() { FG_CUDA_CHECK(cudaEventRecord(start_, stream_)); }
  void stop() { FG_CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

  float elapsed_ms() const {
    FG_CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    FG_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaStream_t stream_;
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

}  // namespace flashgrad::cuda
