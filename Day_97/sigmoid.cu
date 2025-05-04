#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoidKernel(const float* __restrict__ input, float* __restrict__ output, size_t n, size_t m) {

    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // Column index (4 elements)


    if (i < m && j + 3 < n) {
        int idx = i * n + j;
        // Load float4 from input
        float4 in = *reinterpret_cast<const float4*>(&input[idx]);
        float4 out;
        // Apply sigmoid to each component
        out.x = 1.0f / (1.0f + expf(-in.x));
        out.y = 1.0f / (1.0f + expf(-in.y));
        out.z = 1.0f / (1.0f + expf(-in.z));
        out.w = 1.0f / (1.0f + expf(-in.w));
        // Store float4 to output
        *reinterpret_cast<float4*>(&output[idx]) = out;
    }

    else if (i < m && j < n) {
        int idx = i * n + j;
        for (int k = j; k < n && k < j + 4; ++k) {
            output[idx + (k - j)] = 1.0f / (1.0f + expf(-input[idx + (k - j)]));
        }
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {

    dim3 threadsPerBlock(128, 2); 
    dim3 blocksPerGrid(
        ((n + 3) / 4 + threadsPerBlock.x - 1) / threadsPerBlock.x, // ceil((n/4)/32)
        (m + threadsPerBlock.y - 1) / threadsPerBlock.y            // ceil(m/8)
    );

    sigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, n, m);
}