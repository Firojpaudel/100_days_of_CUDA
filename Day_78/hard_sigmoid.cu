#include <cuda_runtime.h>

__global__ void hardSigmoidKernel(
    const float* __restrict__ a,
    float* __restrict__ c,
    size_t m,
    size_t n
) {
    size_t i = blockIdx.y;
    size_t j = (blockIdx.x * blockDim.x + threadIdx.x) * 4; 

    if (i < m && j < n) {
        float4 x = *reinterpret_cast<const float4*>(&a[i * n + j]);

        float4 result;
        result.x = fminf(1.0f, fmaxf(0.0f, (x.x + 3.0f) * 0.16666667f));
        result.y = (j + 1 < n) ? fminf(1.0f, fmaxf(0.0f, (x.y + 3.0f) * 0.16666667f)) : 0.0f;
        result.z = (j + 2 < n) ? fminf(1.0f, fmaxf(0.0f, (x.z + 3.0f) * 0.16666667f)) : 0.0f;
        result.w = (j + 3 < n) ? fminf(1.0f, fmaxf(0.0f, (x.w + 3.0f) * 0.16666667f)) : 0.0f;

        *reinterpret_cast<float4*>(&c[i * n + j]) = result;
    }
}

extern "C" void solution(
    const float* a,
    float* c,
    size_t m,
    size_t n
) {
    dim3 threadsPerBlock(256, 1); 
    dim3 blocksPerGrid((n + threadsPerBlock.x * 4 - 1) / (threadsPerBlock.x * 4), m);
    hardSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, m, n);
}