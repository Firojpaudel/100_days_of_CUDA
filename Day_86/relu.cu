#include <cuda_runtime.h>

__global__ void reluKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t n,
    size_t m
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 4; 

    if (row < n && col < m) {
        size_t idx = row * m + col;
        size_t vec_idx = idx / 4;
        if (col + 3 < m) {
            float4 a_vals = reinterpret_cast<const float4*>(input)[vec_idx];
            float4 c_vals;
            c_vals.x = a_vals.x >= 0.0f ? a_vals.x : 0.0f;
            c_vals.y = a_vals.y >= 0.0f ? a_vals.y : 0.0f;
            c_vals.z = a_vals.z >= 0.0f ? a_vals.z : 0.0f;
            c_vals.w = a_vals.w >= 0.0f ? a_vals.w : 0.0f;
            reinterpret_cast<float4*>(output)[vec_idx] = c_vals;
        } else if (blockIdx.x == (m + 3) / 4 / blockDim.x && threadIdx.x == 0) {
            // Handle remaining elements (m % 4 != 0) with one thread
            for (size_t j = col; j < m; ++j) {
                size_t scalar_idx = row * m + j;
                output[scalar_idx] = __ldg(&input[scalar_idx]) >= 0.0f ? __ldg(&input[scalar_idx]) : 0.0f;
            }
        }
    }
}

extern "C" void solution(
    const float* input,
    float* output,
    size_t n,
    size_t m
) {
    if (n == 0 || m == 0) {
        cudaMemset(output, 0, sizeof(float) * n * m);
        return;
    }

    dim3 blockDim(32, 8);
    dim3 gridDim(((m + 3) / 4 + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    reluKernel<<<gridDim, blockDim>>>(input, output, n, m);
}