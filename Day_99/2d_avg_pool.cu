#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_DIM 16

__global__ void avgPoolKernelOpt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H, int W,
    int kernel_size,
    int stride,
    int padding,
    int H_out, int W_out
) {
    int out_i = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int out_j = blockIdx.x * BLOCK_DIM + threadIdx.x;

    if (out_i >= H_out || out_j >= W_out) return;

    int start_row = out_i * stride - padding;
    int start_col = out_j * stride - padding;

    float sum = 0.0f;
    int pool_area = kernel_size * kernel_size;

    // Optimized unrolling for kernel_size == 3 (common case)
    if (kernel_size == 3 && start_col >= 0 && start_row >= 0 &&
        (start_row + 2) < H && (start_col + 2) < W) {

        #pragma unroll
        for (int m = 0; m < 3; ++m) {
            int in_row = start_row + m;
            const float* row_ptr = input + in_row * W + start_col;
            sum += row_ptr[0] + row_ptr[1] + row_ptr[2];
        }
    } else {
        // Generic case
        for (int m = 0; m < kernel_size; ++m) {
            int in_row = start_row + m;
            if (in_row < 0 || in_row >= H) continue;
            for (int n = 0; n < kernel_size; ++n) {
                int in_col = start_col + n;
                if (in_col >= 0 && in_col < W) {
                    sum += input[in_row * W + in_col];
                }
            }
        }
    }

    output[out_i * W_out + out_j] = sum / pool_area;
}

extern "C" void solution(
    const float* input,
    int kernel_size,
    int stride,
    int padding,
    float* output,
    size_t H,
    size_t W
) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    if (H_out <= 0 || W_out <= 0) {
        cudaMemset(output, 0, sizeof(float) * H_out * W_out);
        return;
    }

    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((W_out + BLOCK_DIM - 1) / BLOCK_DIM, (H_out + BLOCK_DIM - 1) / BLOCK_DIM);

    avgPoolKernelOpt<<<grid, threads>>>(
        input, output,
        (int)H, (int)W,
        kernel_size, stride, padding,
        H_out, W_out
    );
}
