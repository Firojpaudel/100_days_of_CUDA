#include <cuda_runtime.h>

// Constant memory for B (64 KB, supports up to 16384 floats)
__constant__ float c_B[16384];

__global__ void conv2D(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       size_t H, size_t W, size_t Kh, size_t Kw,
                       int use_constant_memory)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < H && col < W) {
        int r_h = (Kh - 1) / 2;
        int r_w = (Kw - 1) / 2;
        float sum = 0.0f;

        // Use #pragma unroll only for small Kh (e.g., <= 7)
        if (Kh <= 7) {
            #pragma unroll
            for (int k_h = 0; k_h < Kh; ++k_h) {
                for (int k_w = 0; k_w < Kw; ++k_w) {
                    int curr_row = row - r_h + k_h;
                    int curr_col = col - r_w + k_w;
                    float b = use_constant_memory ? c_B[k_h * Kw + k_w] : B[k_h * Kw + k_w];
                    if (curr_row >= 0 && curr_row < H && curr_col >= 0 && curr_col < W) {
                        sum += A[curr_row * W + curr_col] * b;
                    }
                }
            }
        } else {
            for (int k_h = 0; k_h < Kh; ++k_h) {
                for (int k_w = 0; k_w < Kw; ++k_w) {
                    int curr_row = row - r_h + k_h;
                    int curr_col = col - r_w + k_w;
                    float b = use_constant_memory ? c_B[k_h * Kw + k_w] : B[k_h * Kw + k_w];
                    if (curr_row >= 0 && curr_row < H && curr_col >= 0 && curr_col < W) {
                        sum += A[curr_row * W + curr_col] * b;
                    }
                }
            }
        }

        C[row * W + col] = sum;
    }
}

extern "C" void solution(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw)
{
    // Use constant memory if Kh * Kw <= 16384
    int use_constant_memory = (Kh * Kw <= 16384) ? 1 : 0;
    if (use_constant_memory) {
        cudaMemcpyToSymbol(c_B, B, Kh * Kw * sizeof(float));
    }

    dim3 blockDim(32, 8); 
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y);

    conv2D<<<gridDim, blockDim>>>(A, B, C, H, W, Kh, Kw, use_constant_memory);
    cudaDeviceSynchronize();
}