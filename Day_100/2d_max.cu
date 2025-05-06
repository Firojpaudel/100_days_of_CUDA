#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h> // For FLT_MIN

#define BLOCK_DIM 16
#define SHARED_TILE (BLOCK_DIM + 2) 

__global__ void maxPoolKernelOpt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H, int W,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int H_out, int W_out
) {
    // Shared memory for input tile
    __shared__ float tile[SHARED_TILE][SHARED_TILE];

    int out_i = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int out_j = blockIdx.x * BLOCK_DIM + threadIdx.x;

    // Calculate base input coordinates for the block
    int base_row = blockIdx.y * BLOCK_DIM * stride - padding;
    int base_col = blockIdx.x * BLOCK_DIM * stride - padding;

    // Load input tile into shared memory (coalesced global memory reads)
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Each thread loads one element (or more for larger tiles)
    for (int i = thread_row; i < SHARED_TILE; i += BLOCK_DIM) {
        for (int j = thread_col; j < SHARED_TILE; j += BLOCK_DIM) {
            int in_row = base_row + i;
            int in_col = base_col + j;
            if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W) {
                tile[i][j] = input[in_row * W + in_col];
            } else {
                tile[i][j] = -FLT_MAX; // Negative infinity for out-of-bounds
            }
        }
    }
    __syncthreads();

    if (out_i >= H_out || out_j >= W_out) return;

    int start_row = out_i * stride - padding - base_row; // Relative to tile
    int start_col = out_j * stride - padding - base_col;

    float max_val = -FLT_MAX;

    // Optimized path for kernel_size == 3 and dilation == 1
    if (kernel_size == 3 && dilation == 1 && 
        start_row >= 0 && start_row + 2 < SHARED_TILE && 
        start_col >= 0 && start_col + 2 < SHARED_TILE) {
        #pragma unroll
        for (int m = 0; m < 3; ++m) {
            #pragma unroll
            for (int n = 0; n < 3; ++n) {
                max_val = fmaxf(max_val, tile[start_row + m][start_col + n]);
            }
        }
    } else {
        // Generic case with dilation
        for (int m = 0; m < kernel_size; ++m) {
            int in_row = start_row + m * dilation;
            if (in_row < 0 || in_row >= SHARED_TILE) continue;
            for (int n = 0; n < kernel_size; ++n) {
                int in_col = start_col + n * dilation;
                if (in_col >= 0 && in_col < SHARED_TILE) {
                    max_val = fmaxf(max_val, tile[in_row][in_col]);
                }
            }
        }
    }

    output[out_i * W_out + out_j] = max_val;
}

extern "C" void solution(
    const float* input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    float* output,
    size_t H,
    size_t W
) {
    // Calculate output dimensions with dilation
    int effective_kernel_size = dilation * (kernel_size - 1) + 1;
    int H_out = (H + 2 * padding - effective_kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - effective_kernel_size) / stride + 1;

    if (H_out <= 0 || W_out <= 0) {
        cudaMemset(output, 0, sizeof(float) * H_out * W_out);
        return;
    }

    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((W_out + BLOCK_DIM - 1) / BLOCK_DIM, (H_out + BLOCK_DIM - 1) / BLOCK_DIM);

    maxPoolKernelOpt<<<grid, threads>>>(
        input, output,
        (int)H, (int)W,
        kernel_size, stride, padding, dilation,
        H_out, W_out
    );
}