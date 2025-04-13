#include <cuda_runtime.h>

__global__ void cumProdNaiveKernel(const float* input, float* output, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        float prod = 1.0f;
        for (size_t j = 0; j <= i; ++j) {
            prod *= input[j];
        }
        output[i] = prod;
    }
}

extern "C" void solution(const float* input, float* output, size_t n) {
    dim3 blockDim(256); // Simple 1D block, 256 threads
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
    cumProdNaiveKernel<<<gridDim, blockDim>>>(input, output, n);
}