#include <cuda_runtime.h>

__global__ void conv2d(const float* A,
                       const float* B,
                       float* C,
                       size_t H,
                       size_t W,
                       size_t KH,
                       size_t KW)
{
    // Get output position (i, j)
    size_t i = blockIdx.y * blockDim.y + threadIdx.y; // Row
    size_t j = blockIdx.x * blockDim.x + threadIdx.x; // Column

    // Compute only if within output bounds
    if (i < H && j < W) {
        float sum = 0.0f;
        int radius_h = KH / 2; // ⌊KH/2⌋
        int radius_w = KW / 2; // ⌊KW/2⌋

        // Double loop over kernel (k, l)
        for (int k = 0; k < KH; ++k) {
            for (int l = 0; l < KW; ++l) {
                // Input index with zero-padding
                int ai = i + k - radius_h; // Row index in A
                int aj = j + l - radius_w; // Column index in A
                if (ai >= 0 && ai < H && aj >= 0 && aj < W) {
                    // Row-major access: A[ai*W + aj], B[k*KW + l]
                    sum += A[ai * W + aj] * B[k * KW + l];
                }
                // Else: zero-padding (skip multiplication)
            }
        }
        // Write to output: C[i*W + j]
        C[i * W + j] = sum;
    }
}

extern "C" void solution(const float* A,
                         const float* B,
                         float* C,
                         size_t H,
                         size_t W,
                         size_t KH,
                         size_t KW)
{
    // Simple 2D block and grid configuration
    dim3 threads(16, 16); // 16x16 threads per block (256 threads)
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    conv2d<<<blocks, threads>>>(A, B, C, H, W, KH, KW);
}