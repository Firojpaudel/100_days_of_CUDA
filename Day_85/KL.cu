#include <cuda_runtime.h>

__const__ float epsilon = 1e-10f;

__global__ void kullback_divergence(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    size_t N
) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx < N) {
        // Load 4 elements with float4
        float4 pred, target;
        if (idx + 3 < N) {
            pred = *reinterpret_cast<const float4*>(&predictions[idx]);
            target = *reinterpret_cast<const float4*>(&targets[idx]);
        } else {
            pred.x = idx < N ? __ldg(&predictions[idx]) : 0.0f;
            pred.y = idx + 1 < N ? __ldg(&predictions[idx + 1]) : 0.0f;
            pred.z = idx + 2 < N ? __ldg(&predictions[idx + 2]) : 0.0f;
            pred.w = idx + 3 < N ? __ldg(&predictions[idx + 3]) : 0.0f;
            target.x = idx < N ? __ldg(&targets[idx]) : 0.0f;
            target.y = idx + 1 < N ? __ldg(&targets[idx + 1]) : 0.0f;
            target.z = idx + 2 < N ? __ldg(&targets[idx + 2]) : 0.0f;
            target.w = idx + 3 < N ? __ldg(&targets[idx + 3]) : 0.0f;
        }

        // Compute KL divergence for 4 elements
        float4 out;
        out.x = target.x * (__logf(target.x + epsilon) - __logf(pred.x + epsilon));
        out.y = idx + 1 < N ? target.y * (__logf(target.y + epsilon) - __logf(pred.y + epsilon)) : 0.0f;
        out.z = idx + 2 < N ? target.z * (__logf(target.z + epsilon) - __logf(pred.z + epsilon)) : 0.0f;
        out.w = idx + 3 < N ? target.w * (__logf(target.w + epsilon) - __logf(pred.w + epsilon)) : 0.0f;

        // Store 4 elements with float4
        if (idx + 3 < N) {
            *reinterpret_cast<float4*>(&output[idx]) = out;
        } else {
            if (idx < N) output[idx] = out.x;
            if (idx + 1 < N) output[idx + 1] = out.y;
            if (idx + 2 < N) output[idx + 2] = out.z;
            if (idx + 3 < N) output[idx + 3] = out.w;
        }
    }
}

extern "C" void solution(
    const float* predictions,
    const float* targets,
    float* output,
    size_t n
) {
    int threads = 256;
    int blocks = (n + threads * 4 - 1) / (threads * 4);

    kullback_divergence<<<blocks, threads>>>(predictions, targets, output, n);

    cudaDeviceSynchronize();
}