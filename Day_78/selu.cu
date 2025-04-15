#include <cuda_runtime.h>

__global__ void seluKernel(
    const float* __restrict__ a,
    float* __restrict__ c,
    size_t m,
    size_t n
) {
    size_t i = blockIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j * 4 < n) {
        float4 x = *reinterpret_cast<const float4*>(&a[i * n + j * 4]);

        const float lambda = 1.050701f;
        const float alpha = 1.673263f;

        float4 result;
        result.x = lambda * (fmaxf(0.0f, x.x) + fminf(0.0f, alpha * (expf(x.x) - 1.0f)));
        result.y = (j * 4 + 1 < n) ? lambda * (fmaxf(0.0f, x.y) + fminf(0.0f, alpha * (expf(x.y) - 1.0f))) : 0.0f;
        result.z = (j * 4 + 2 < n) ? lambda * (fmaxf(0.0f, x.z) + fminf(0.0f, alpha * (expf(x.z) - 1.0f))) : 0.0f;
        result.w = (j * 4 + 3 < n) ? lambda * (fmaxf(0.0f, x.w) + fminf(0.0f, alpha * (expf(x.w) - 1.0f))) : 0.0f;

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
    seluKernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, m, n);
}