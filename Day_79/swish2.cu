#include <cuda_runtime.h>

__global__ void swishKernel(
    const float* __restrict__ a,
    float* __restrict__ c,
    size_t m,
    size_t n
) {
    size_t i = blockIdx.y;
    size_t j = (blockIdx.x * blockDim.x + threadIdx.x) * 8; // 8 elements/thread

    if (i < m && j < n) {
        float4 x1, x2;
        if (j + 3 < n) {
            x1 = *reinterpret_cast<const float4*>(&a[i * n + j]);
        } else {
            x1.x = (j + 0 < n) ? a[i * n + j + 0] : 0.0f;
            x1.y = (j + 1 < n) ? a[i * n + j + 1] : 0.0f;
            x1.z = (j + 2 < n) ? a[i * n + j + 2] : 0.0f;
            x1.w = (j + 3 < n) ? a[i * n + j + 3] : 0.0f;
        }
        if (j + 7 < n) {
            x2 = *reinterpret_cast<const float4*>(&a[i * n + j + 4]);
        } else {
            x2.x = (j + 4 < n) ? a[i * n + j + 4] : 0.0f;
            x2.y = (j + 5 < n) ? a[i * n + j + 5] : 0.0f;
            x2.z = (j + 6 < n) ? a[i * n + j + 6] : 0.0f;
            x2.w = (j + 7 < n) ? a[i * n + j + 7] : 0.0f;
        }

        float4 result1, result2;
        result1.x = x1.x / (1.0f + expf(-x1.x));
        result1.y = x1.y / (1.0f + expf(-x1.y));
        result1.z = x1.z / (1.0f + expf(-x1.z));
        result1.w = x1.w / (1.0f + expf(-x1.w));
        result2.x = x2.x / (1.0f + expf(-x2.x));
        result2.y = x2.y / (1.0f + expf(-x2.y));
        result2.z = x2.z / (1.0f + expf(-x2.z));
        result2.w = x2.w / (1.0f + expf(-x2.w));

        if (j + 3 < n) {
            *reinterpret_cast<float4*>(&c[i * n + j]) = result1;
        } else {
            if (j + 0 < n) c[i * n + j + 0] = result1.x;
            if (j + 1 < n) c[i * n + j + 1] = result1.y;
            if (j + 2 < n) c[i * n + j + 2] = result1.z;
            if (j + 3 < n) c[i * n + j + 3] = result1.w;
        }
        if (j + 7 < n) {
            *reinterpret_cast<float4*>(&c[i * n + j + 4]) = result2;
        } else {
            if (j + 4 < n) c[i * n + j + 4] = result2.x;
            if (j + 5 < n) c[i * n + j + 5] = result2.y;
            if (j + 6 < n) c[i * n + j + 6] = result2.z;
            if (j + 7 < n) c[i * n + j + 7] = result2.w;
        }
    }
}

extern "C" void solution(
    const float* a,
    float* c,
    size_t m,
    size_t n
) {
    dim3 threadsPerBlock(128, 1);
    dim3 blocksPerGrid((n + threadsPerBlock.x * 8 - 1) / (threadsPerBlock.x * 8), m);
    swishKernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, m, n);
}