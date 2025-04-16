#include <cuda_runtime.h>

__global__ void swishKernel(
    const float* __restrict__ a,
    float* __restrict__ c,
    size_t m,
    size_t n
) {
    size_t i = blockIdx.y;
    size_t j = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (i < m && j < n) {
        float4 x;
        if (j + 3 < n) {
            x = *reinterpret_cast<const float4*>(&a[i * n + j]);
        } else {
            x.x = (j + 0 < n) ? a[i * n + j + 0] : 0.0f;
            x.y = (j + 1 < n) ? a[i * n + j + 1] : 0.0f;
            x.z = (j + 2 < n) ? a[i * n + j + 2] : 0.0f;
            x.w = (j + 3 < n) ? a[i * n + j + 3] : 0.0f;
        }

        float4 result;
        result.x = x.x / (1.0f + __expf(-x.x));
        result.y = x.y / (1.0f + __expf(-x.y));
        result.z = x.z / (1.0f + __expf(-x.z));
        result.w = x.w / (1.0f + __expf(-x.w));

        if (j + 3 < n) {
            *reinterpret_cast<float4*>(&c[i * n + j]) = result;
        } else {
            if (j + 0 < n) c[i * n + j + 0] = result.x;
            if (j + 1 < n) c[i * n + j + 1] = result.y;
            if (j + 2 < n) c[i * n + j + 2] = result.z;
            if (j + 3 < n) c[i * n + j + 3] = result.w;
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
    dim3 blocksPerGrid((n + threadsPerBlock.x * 4 - 1) / (threadsPerBlock.x * 4), m);
    swishKernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, m, n);
}