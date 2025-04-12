#include <cuda_runtime.h>

__global__ void eluKernel(const float* input, float* output, size_t n, size_t m, float alpha) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // Process 4 floats
    
    if (row < n && col < m) {
        size_t idx = row * m + col;
        
        // Process 4 floats with float4
        if (col + 3 < m) {
            float4 a_vals = *reinterpret_cast<const float4*>(&input[idx]);
            float4 c_vals;
            c_vals.x = a_vals.x >= 0.0f ? a_vals.x : alpha * (expf(a_vals.x) - 1.0f);
            c_vals.y = a_vals.y >= 0.0f ? a_vals.y : alpha * (expf(a_vals.y) - 1.0f);
            c_vals.z = a_vals.z >= 0.0f ? a_vals.z : alpha * (expf(a_vals.z) - 1.0f);
            c_vals.w = a_vals.w >= 0.0f ? a_vals.w : alpha * (expf(a_vals.w) - 1.0f);
            *reinterpret_cast<float4*>(&output[idx]) = c_vals;
        } else {
            #pragma unroll
            for (size_t i = 0; i < 4 && (col + i) < m; ++i) {
                float x = input[idx + i];
                output[idx + i] = x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
            }
        }
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    dim3 blockDim(32, 8); // 256 threads
    dim3 gridDim(((m + 3) / 4 + blockDim.x - 1) / blockDim.x, 
                 (n + blockDim.y - 1) / blockDim.y);
    eluKernel<<<gridDim, blockDim>>>(input, output, n, m, alpha);
}