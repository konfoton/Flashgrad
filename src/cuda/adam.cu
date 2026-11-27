#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "cuda/memory.h"
#include "cuda/ops.h"
#include "flashgrad/core/error.h"
#include "flashgrad/cuda/check.h"

// Fused Adam / AdamW step with multi-tensor apply.
//
// The unfused step runs about a dozen elementwise kernels per parameter, each reading and
// writing whole tensors (and allocating temporaries). Here one kernel does the whole update
// per element in registers, for many parameters at once: the tensors' pointers travel in the
// kernel arguments, every tensor is cut into chunks of kChunkSize elements, and each block
// handles one chunk. Per element that is 4 loads and 3 stores, the minimum for Adam.
namespace flashgrad::cuda::detail {

namespace {

constexpr int kThreads = 256;
constexpr std::int64_t kChunkSize = 4096;  // elements per block; a multiple of 4 (float4)
// Kernel arguments are limited to 4 KB; TensorList below is about 3 KB with 64 tensors.
constexpr int kMaxTensorsPerLaunch = 64;

// The tensors of one launch. chunk_begin[t] is the index of the first block of tensor t, so
// tensor t is handled by blocks [chunk_begin[t], chunk_begin[t + 1]).
struct TensorList {
  float* param[kMaxTensorsPerLaunch];
  const float* grad[kMaxTensorsPerLaunch];
  float* exp_avg[kMaxTensorsPerLaunch];
  float* exp_avg_sq[kMaxTensorsPerLaunch];
  std::int64_t numel[kMaxTensorsPerLaunch];
  std::int64_t chunk_begin[kMaxTensorsPerLaunch + 1];
  int count;
};
static_assert(sizeof(TensorList) < 4000, "TensorList must fit in the kernel arguments");

// The step's constants, folded on the host in double.
struct Coefficients {
  float beta1;
  float beta2;
  float eps;
  float step_size;              // lr / (1 - beta1^t)
  float sqrt_bias_correction2;  // sqrt(1 - beta2^t)
  float l2;                     // weight_decay for Adam, 0 for AdamW
  float param_scale;            // 1 - lr * weight_decay for AdamW, 1 for Adam
};

__device__ __forceinline__ void adam_update(float& p, float g, float& m, float& v,
                                            const Coefficients& c) {
  // Branches on uniform values, so they don't diverge; skipping them when they're no-ops
  // keeps inf/NaN parameters from turning into NaN through 0 * p.
  if (c.param_scale != 1.0f) p *= c.param_scale;
  if (c.l2 != 0.0f) g = fmaf(c.l2, p, g);
  m = fmaf(c.beta1, m, (1.0f - c.beta1) * g);
  v = fmaf(c.beta2, v, (1.0f - c.beta2) * g * g);
  const float denominator = sqrtf(v) / c.sqrt_bias_correction2 + c.eps;
  p -= c.step_size * m / denominator;
}

// kVectorized: every pointer is 16-byte aligned, so whole float4s are loaded and stored
// (chunks start at multiples of kChunkSize, which keeps them aligned).
template <bool kVectorized>
__global__ void __launch_bounds__(kThreads) adam_kernel(TensorList list, Coefficients c) {
  // The tensor of this block: the last t with chunk_begin[t] <= blockIdx.x. Every thread
  // reads the same entries, so the reads from the argument space are broadcast.
  const auto block = static_cast<std::int64_t>(blockIdx.x);
  int lo = 0, hi = list.count - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (list.chunk_begin[mid] <= block)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int t = lo;
  float* __restrict__ param = list.param[t];
  const float* __restrict__ grad = list.grad[t];
  float* __restrict__ exp_avg = list.exp_avg[t];
  float* __restrict__ exp_avg_sq = list.exp_avg_sq[t];
  const std::int64_t begin = (block - list.chunk_begin[t]) * kChunkSize;
  const std::int64_t end = begin + kChunkSize < list.numel[t] ? begin + kChunkSize : list.numel[t];

  std::int64_t scalar_begin = begin;
  if constexpr (kVectorized) {
    const std::int64_t vector_end = begin + (end - begin) / 4 * 4;
    for (std::int64_t i = begin + 4 * static_cast<std::int64_t>(threadIdx.x); i < vector_end;
         i += 4 * static_cast<std::int64_t>(blockDim.x)) {
      float4 p = *reinterpret_cast<const float4*>(param + i);
      const float4 g = *reinterpret_cast<const float4*>(grad + i);
      float4 m = *reinterpret_cast<const float4*>(exp_avg + i);
      float4 v = *reinterpret_cast<const float4*>(exp_avg_sq + i);
      adam_update(p.x, g.x, m.x, v.x, c);
      adam_update(p.y, g.y, m.y, v.y, c);
      adam_update(p.z, g.z, m.z, v.z, c);
      adam_update(p.w, g.w, m.w, v.w, c);
      *reinterpret_cast<float4*>(param + i) = p;
      *reinterpret_cast<float4*>(exp_avg + i) = m;
      *reinterpret_cast<float4*>(exp_avg_sq + i) = v;
    }
    scalar_begin = vector_end;  // the last < 4 elements of the tensor
  }
  for (std::int64_t i = scalar_begin + threadIdx.x; i < end; i += blockDim.x) {
    float p = param[i];
    float m = exp_avg[i];
    float v = exp_avg_sq[i];
    adam_update(p, grad[i], m, v, c);
    param[i] = p;
    exp_avg[i] = m;
    exp_avg_sq[i] = v;
  }
}

bool is_aligned16(const void* ptr) { return reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0; }

Coefficients make_coefficients(const AdamStep& s) {
  const double t = static_cast<double>(s.step);
  const double bias_correction1 = 1.0 - std::pow(static_cast<double>(s.beta1), t);
  const double bias_correction2 = 1.0 - std::pow(static_cast<double>(s.beta2), t);
  Coefficients c{};
  c.beta1 = s.beta1;
  c.beta2 = s.beta2;
  c.eps = s.eps;
  c.step_size = static_cast<float>(static_cast<double>(s.lr) / bias_correction1);
  c.sqrt_bias_correction2 = static_cast<float>(std::sqrt(bias_correction2));
  c.l2 = s.decoupled_weight_decay ? 0.0f : s.weight_decay;
  c.param_scale = s.decoupled_weight_decay
                      ? static_cast<float>(1.0 - static_cast<double>(s.lr) * s.weight_decay)
                      : 1.0f;
  return c;
}

}  // namespace

void adam_step(const std::vector<AdamTensors>& tensors, const AdamStep& step, Stream stream) {
  if (tensors.empty()) return;
  FG_CHECK(step.step >= 1, "adam_step: step must be >= 1, got ", step.step);
  DeviceGuard guard(tensors.front().param.device().index());
  const Coefficients c = make_coefficients(step);

  TensorList list{};
  std::int64_t chunks = 0;
  bool aligned = true;
  const auto launch = [&] {
    if (list.count == 0) return;
    list.chunk_begin[list.count] = chunks;
    FG_CHECK(chunks <= kMaxBlocks, "adam_step: ", chunks, " chunks exceed the grid limit");
    const auto num_blocks = static_cast<std::uint32_t>(chunks);
    if (aligned)
      adam_kernel<true><<<num_blocks, kThreads, 0, stream>>>(list, c);
    else
      adam_kernel<false><<<num_blocks, kThreads, 0, stream>>>(list, c);
    FG_CUDA_KERNEL_CHECK();
    list = TensorList{};
    chunks = 0;
    aligned = true;
  };

  for (const AdamTensors& set : tensors) {
    const std::int64_t n = set.param.numel();
    if (n == 0) continue;
    const int i = list.count;
    list.param[i] = set.param.data_ptr();
    list.grad[i] = set.grad.data_ptr();
    list.exp_avg[i] = set.exp_avg.data_ptr();
    list.exp_avg_sq[i] = set.exp_avg_sq.data_ptr();
    list.numel[i] = n;
    list.chunk_begin[i] = chunks;
    chunks += (n + kChunkSize - 1) / kChunkSize;
    aligned = aligned && is_aligned16(list.param[i]) && is_aligned16(list.grad[i]) &&
              is_aligned16(list.exp_avg[i]) && is_aligned16(list.exp_avg_sq[i]);
    if (++list.count == kMaxTensorsPerLaunch) launch();
  }
  launch();
}

}  // namespace flashgrad::cuda::detail
