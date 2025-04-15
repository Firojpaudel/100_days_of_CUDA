#include <cuda_runtime.h>

__global__ void huberLossKernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i * 4 < n) {
        float4 p = *reinterpret_cast<const float4*>(&predictions[i * 4]);
        float4 t = *reinterpret_cast<const float4*>(&targets[i * 4]);

        float4 result;
        float diff_x = p.x - t.x;
        float abs_diff_x = fabsf(diff_x);
        result.x = (abs_diff_x <= 1.0f) ? 0.5f * diff_x * diff_x : abs_diff_x - 0.5f;

        float diff_y = p.y - t.y;
        float abs_diff_y = fabsf(diff_y);
        result.y = (i * 4 + 1 < n) ? ((abs_diff_y <= 1.0f) ? 0.5f * diff_y * diff_y : abs_diff_y - 0.5f) : 0.0f;

        float diff_z = p.z - t.z;
        float abs_diff_z = fabsf(diff_z);
        result.z = (i * 4 + 2 < n) ? ((abs_diff_z <= 1.0f) ? 0.5f * diff_z * diff_z : abs_diff_z - 0.5f) : 0.0f;

        float diff_w = p.w - t.w;
        float abs_diff_w = fabsf(diff_w);
        result.w = (i * 4 + 3 < n) ? ((abs_diff_w <= 1.0f) ? 0.5f * diff_w * diff_w : abs_diff_w - 0.5f) : 0.0f;

        *reinterpret_cast<float4*>(&output[i * 4]) = result;
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
    huberLossKernel<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, output, n);
}