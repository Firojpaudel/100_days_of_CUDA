#include <cuda_runtime.h>

__constant__ float epsilon = 1e-10f;

__global__ void kullback_divergence(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    size_t N
) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx >= N) return; // Early exit for out-of-bounds threads

    // Load 4 elements with float4, handle boundaries
    float4 pred, target;
    pred.x = idx < N ? __ldg(&predictions[idx]) : 1.0f;
    pred.y = idx + 1 < N ? __ldg(&predictions[idx + 1]) : 1.0f;
    pred.z = idx + 2 < N ? __ldg(&predictions[idx + 2]) : 1.0f;
    pred.w = idx + 3 < N ? __ldg(&predictions[idx + 3]) : 1.0f;
    target.x = idx < N ? __ldg(&targets[idx]) : 0.0f;
    target.y = idx + 1 < N ? __ldg(&targets[idx + 1]) : 0.0f;
    target.z = idx + 2 < N ? __ldg(&targets[idx + 2]) : 0.0f;
    target.w = idx + 3 < N ? __ldg(&targets[idx + 3]) : 0.0f;

    // Compute KL divergence with single log
    float4 out;
    out.x = idx < N ? target.x * __logf((target.x + epsilon) / (pred.x + epsilon)) : 0.0f;
    out.y = idx + 1 < N ? target.y * __logf((target.y + epsilon) / (pred.y + epsilon)) : 0.0f;
    out.z = idx + 2 < N ? target.z * __logf((target.z + epsilon) / (pred.z + epsilon)) : 0.0f;
    out.w = idx + 3 < N ? target.w * __logf((target.w + epsilon) / (pred.w + epsilon)) : 0.0f;

    // Store 4 elements with float4, only if within bounds
    if (idx < N) {
        *reinterpret_cast<float4*>(&output[idx]) = out;
    }
}

extern "C" void solution(
    const float* predictions,
    const float* targets,
    float* output,
    size_t n
) {
    if (n == 0) return;

    // Dynamic thread block size
    int threads = 256;
    if (n < 256 * 4) threads = 128; // Smaller blocks for small n
    else if (n > 1024 * 1024) threads = 512; // Larger blocks for large n

    int blocks = (n + threads * 4 - 1) / (threads * 4);

    kullback_divergence<<<blocks, threads>>>(predictions, targets, output, n);
}