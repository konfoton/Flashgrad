#pragma once

#include <vector>

// Minimal end-to-end GPU path (host -> device -> kernel -> host) used to
// verify that the toolchain, the build and the driver all work.
// Will be superseded by real tensor ops in Phase 1/2.
namespace flashgrad::cuda::smoke {

// out[i] = a[i] + b[i], computed on the current CUDA device.
std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b);

}  // namespace flashgrad::cuda::smoke
