#include <cuda_runtime.h>

__global__ void swishKernel(
    const float* __restrict__ a,
    float* __restrict__ c,
    size_t m,
    size_t n
) {
    size_t i = blockIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j * 4 < n) {
        float4 x = *reinterpret_cast<const float4*>(&a[i * n + j * 4]);

        float4 result;
        result.x = x.x / (1.0f + expf(-x.x));
        result.y = (j * 4 + 1 < n) ? x.y / (1.0f + expf(-x.y)) : 0.0f;
        result.z = (j * 4 + 2 < n) ? x.z / (1.0f + expf(-x.z)) : 0.0f;
        result.w = (j * 4 + 3 < n) ? x.w / (1.0f + expf(-x.w)) : 0.0f;

        *reinterpret_cast<float4*>(&c[i * n + j * 4]) = result;
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
    swishKernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, m, n);
}