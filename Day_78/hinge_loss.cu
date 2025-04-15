#include <cuda_runtime.h>

__global__ void hingeLossKernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Process 4 elements per thread with float4
    if (i * 4 < n) {
        float4 p = *reinterpret_cast<const float4*>(&predictions[i * 4]);
        float4 t = *reinterpret_cast<const float4*>(&targets[i * 4]);

        output[i * 4]     = fmaxf(0.0f, 1.0f - p.x * t.x);
        if (i * 4 + 1 < n) output[i * 4 + 1] = fmaxf(0.0f, 1.0f - p.y * t.y);
        if (i * 4 + 2 < n) output[i * 4 + 2] = fmaxf(0.0f, 1.0f - p.z * t.z);
        if (i * 4 + 3 < n) output[i * 4 + 3] = fmaxf(0.0f, 1.0f - p.w * t.w);
    }
}

extern "C" void solution(
    const float* predictions,
    const float* targets,
    float* output,
    size_t n
) {
    int threadsPerBlock = 256; 
    int blocksPerGrid = (n + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
    hingeLossKernel<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, output, n);
}