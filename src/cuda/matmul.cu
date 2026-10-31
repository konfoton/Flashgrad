#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "cuda/memory.h"
#include "cuda/ops.h"
#include "flashgrad/core/error.h"
#include "flashgrad/cuda/check.h"

// Matrix multiplication: shared-memory tiling, a TM x TN register tile per thread, 
// A transposed in shared memory and float4 global loads/stores.
//
// The kernel needs row-major operands whose rows are padded with zeros to a multiple of 4
// floats (lda/ldb/ldc) and 16-byte aligned. Inputs already in that layout are used in place;
// others are first copied into a padded buffer: with cudaMemcpy2D when the rows are
// contiguous, otherwise with the identity map kernel. The output goes straight into `out`
// when N is a multiple of 4, otherwise through a padded buffer.
namespace flashgrad::cuda::detail {

namespace {

constexpr int kBM = 64;
constexpr int kBK = 8;
constexpr int kBN = 64;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kThreads = (kBM * kBN) / (kTM * kTN);
constexpr std::size_t kSharedBytes = sizeof(float) * (kBM * kBK + kBK * kBN);

template<int BM, int BK, int BN, int TM, int TN>
__global__ void matmul_v6(const float *A, const float *B, float *C, const int M, const int N, const int K,
                          const int lda, const int ldb, const int ldc)
{
    // lda/ldb/ldc: row strides (in floats) of A/B/C, multiples of 4 so every row is float4-aligned
    extern __shared__ char shm[];
    float *shmA = (float *)(shm);
    float *shmB = (float *)(shm + (BM*BK) * sizeof(float));

    // each warp calcs a TM*TN*32 values (each thread TM*TN vals)
    uint threadRow = threadIdx.x / (BN / TN); // [0 ... (BM / TM)-1]
    uint threadCol = threadIdx.x % (BN / TN);

    uint threadsPerBlock = (BM * BN) / (TM * TN);
    uint strideA = (threadsPerBlock / (BK/4));
    uint strideB = (threadsPerBlock / (BN/4));

    uint blockRow = blockIdx.x / ((N + BN - 1) / BN);
    uint blockCol = blockIdx.x % ((N + BN - 1) / BN);

    A += blockRow * BM * lda;                       // row=blockRow,col=0
    B += blockCol * BN;                             // row=0,col=blockCol
    C += blockRow * BM * ldc + blockCol * BN;       // row=blockRow,col=blockCol

    uint innerRowA = threadIdx.x / (BK/4); // row within the block
    uint innerColA = threadIdx.x % (BK/4); // col within the block
    uint innerRowB = threadIdx.x / (BN/4); // row within the block
    uint innerColB = threadIdx.x % (BN/4); // col within the block

    float threadRegisterResults[TM*TN] = {0.0};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (uint bkIdx=0; bkIdx < K; bkIdx += BK)
    {
        for(uint load_offset = 0; load_offset < BM; load_offset += strideA)
        {
            if ((blockRow * BM + innerRowA + load_offset) < M && (bkIdx + innerColA * 4 + 3) < lda)
            {
                const float4 tmp = 
                    reinterpret_cast<const float4 *>(&A[(innerRowA + load_offset) * lda + innerColA * 4])[0];
                shmA[(innerColA * 4 + 0) * BM + innerRowA + load_offset] = tmp.x;
                shmA[(innerColA * 4 + 1) * BM + innerRowA + load_offset] = tmp.y;
                shmA[(innerColA * 4 + 2) * BM + innerRowA + load_offset] = tmp.z;
                shmA[(innerColA * 4 + 3) * BM + innerRowA + load_offset] = tmp.w;
            }
            else
            {
                shmA[(innerColA * 4 + 0) * BM + innerRowA + load_offset] = 0.0f;
                shmA[(innerColA * 4 + 1) * BM + innerRowA + load_offset] = 0.0f;
                shmA[(innerColA * 4 + 2) * BM + innerRowA + load_offset] = 0.0f;
                shmA[(innerColA * 4 + 3) * BM + innerRowA + load_offset] = 0.0f;
            }
        }

        for(uint load_offset = 0; load_offset < BK; load_offset += strideB)
        {
            if ((bkIdx + innerRowB + load_offset) < K && (blockCol * BN + innerColB * 4 + 3) < ldb)
            {
                reinterpret_cast<float4 *>(&shmB[(innerRowB + load_offset) * BN + innerColB * 4])[0] = 
                    reinterpret_cast<const float4 *>(&B[(innerRowB + load_offset) * ldb + innerColB * 4])[0];
            }
            else
            {
                shmB[(innerRowB + load_offset) * BN + innerColB * 4 + 0] = 0.0f;
                shmB[(innerRowB + load_offset) * BN + innerColB * 4 + 1] = 0.0f;
                shmB[(innerRowB + load_offset) * BN + innerColB * 4 + 2] = 0.0f;
                shmB[(innerRowB + load_offset) * BN + innerColB * 4 + 3] = 0.0f;
            }
        }

        __syncthreads();

        // advance pointers
        A += BK;
        B += BK * ldb;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            for(uint i = 0; i < TM; ++i)
                regM[i] = shmA[dotIdx * BM + threadRow * TM + i];
            for(uint i = 0; i < TN; ++i)
                regN[i] = shmB[dotIdx * BN + threadCol * TN + i];
            for(uint resIdx_M = 0; resIdx_M < TM; ++resIdx_M)
            {
                for(uint resIdx_N = 0; resIdx_N < TN; ++resIdx_N)
                {
                    threadRegisterResults[resIdx_M * TN + resIdx_N] += regM[resIdx_M] * regN[resIdx_N];
                }
            }
        }
        __syncthreads();
    }

    for(uint resIdx_M = 0; resIdx_M < TM; ++resIdx_M)
    {
        uint globalRow = blockRow * BM + threadRow * TM + resIdx_M;
        if (globalRow >= M) break;

        for(uint resIdx_N = 0; resIdx_N < TN; resIdx_N += 4)
        {
            // columns in [N, ldc) are padding, never copied back to host
            uint globalCol = blockCol * BN + threadCol * TN + resIdx_N;
            if (globalCol >= ldc) break;

            float4 out;
            out.x = threadRegisterResults[resIdx_M * TN + resIdx_N + 0];
            out.y = threadRegisterResults[resIdx_M * TN + resIdx_N + 1];
            out.z = threadRegisterResults[resIdx_M * TN + resIdx_N + 2];
            out.w = threadRegisterResults[resIdx_M * TN + resIdx_N + 3];
            reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdx_M) * ldc + threadCol * TN + resIdx_N])[0] = out;
        }
    }
}

std::int64_t padded_ld(std::int64_t cols) { return (cols + 3) / 4 * 4; }

bool is_aligned(const float* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// A [rows, cols] operand in the kernel's layout: row-major with `ld` floats per row.
struct Operand {
  Tensor buffer;  // the padded copy; undefined when the input is used in place
  const float* data;
  std::int64_t ld;
};

Operand pad(const Tensor& t, Stream stream) {
  const std::int64_t rows = t.size(0), cols = t.size(1);
  if (t.is_contiguous() && cols % 4 == 0 && is_aligned(t.data_ptr())) {
    return {Tensor(), t.data_ptr(), cols};
  }
  const std::int64_t ld = padded_ld(cols);
  Tensor buffer = empty({rows, ld}, t.options());
  // padding must be zero so it contributes nothing to the dot products
  FG_CUDA_CHECK(cudaMemsetAsync(buffer.data_ptr(), 0, buffer.nbytes(), stream));
  if (t.numel() == 0) return {buffer, buffer.data_ptr(), ld};
  if (t.stride(1) == 1 && t.stride(0) >= cols) {
    FG_CUDA_CHECK(cudaMemcpy2DAsync(buffer.data_ptr(), ld * sizeof(float), t.data_ptr(),
                                    t.stride(0) * sizeof(float), cols * sizeof(float), rows,
                                    cudaMemcpyDeviceToDevice, stream));
  } else {
    unary(UnaryOp::Identity, buffer.slice(1, 0, cols), t, stream);  // strided copy
  }
  return {buffer, buffer.data_ptr(), ld};
}

}  // namespace

void matmul(const Tensor& out, const Tensor& a, const Tensor& b, Stream stream) {
  const std::int64_t m = a.size(0), k = a.size(1), n = b.size(1);
  if (m == 0 || n == 0) return;
  // The kernel takes int sizes and computes offsets in 32-bit unsigned arithmetic.
  constexpr std::int64_t kMaxInt = std::numeric_limits<int>::max();
  constexpr std::int64_t kMaxOffset = std::numeric_limits<std::uint32_t>::max();
  FG_CHECK(m <= kMaxInt && padded_ld(k) <= kMaxInt && padded_ld(n) <= kMaxInt &&
               m * padded_ld(k) <= kMaxOffset && k * padded_ld(n) <= kMaxOffset &&
               m * padded_ld(n) <= kMaxOffset,
           "matmul: matrices too large for 32-bit indexing, got ", describe(a), " and ",
           describe(b));
  const std::int64_t num_blocks = ((m + kBM - 1) / kBM) * ((n + kBN - 1) / kBN);
  DeviceGuard guard(out.device().index());

  const Operand pa = pad(a, stream);
  const Operand pb = pad(b, stream);
  const bool direct = n % 4 == 0 && is_aligned(out.data_ptr());
  const std::int64_t ldc = direct ? n : padded_ld(n);
  const Tensor c = direct ? out : empty({m, ldc}, out.options());

  matmul_v6<kBM, kBK, kBN, kTM, kTN>
      <<<static_cast<std::uint32_t>(num_blocks), kThreads, kSharedBytes, stream>>>(
          pa.data, pb.data, c.data_ptr(), static_cast<int>(m), static_cast<int>(n),
          static_cast<int>(k), static_cast<int>(pa.ld), static_cast<int>(pb.ld),
          static_cast<int>(ldc));
  FG_CUDA_KERNEL_CHECK();

  if (!direct) {
    // columns in [N, ldc) are padding, not copied to the output
    FG_CUDA_CHECK(cudaMemcpy2DAsync(out.data_ptr(), n * sizeof(float), c.data_ptr(),
                                    ldc * sizeof(float), n * sizeof(float), m,
                                    cudaMemcpyDeviceToDevice, stream));
  }
}

}  // namespace flashgrad::cuda::detail
