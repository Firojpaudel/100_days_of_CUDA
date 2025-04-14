#include <cuda_runtime.h>

__global__ void cosineSimilarityKernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    size_t n,
    size_t d
) {
    // Shared memory for partial sums
    __shared__ float sh_dot[64];
    __shared__ float sh_norm_pred[64];
    __shared__ float sh_norm_targ[64];

    size_t i = blockIdx.x; // One block per vector i
    if (i >= n) return;

    size_t tid = threadIdx.x;
    float dot = 0.0f;
    float norm_pred = 0.0f;
    float norm_targ = 0.0f;

    // Process D in chunks
    for (size_t j = tid * 4; j < d; j += blockDim.x * 4) {
        float4 p = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 t = {0.0f, 0.0f, 0.0f, 0.0f};

        // Load float4, check bounds
        if (j + 3 < d) {
            p = *reinterpret_cast<const float4*>(&predictions[i * d + j]);
            t = *reinterpret_cast<const float4*>(&targets[i * d + j]);
        } else {
            // Handle remainder scalar
            for (size_t k = 0; k < 4 && j + k < d; ++k) {
                p.x = (k == 0) ? predictions[i * d + j] : p.x;
                p.y = (k == 1) ? predictions[i * d + j + 1] : p.y;
                p.z = (k == 2) ? predictions[i * d + j + 2] : p.z;
                p.w = (k == 3) ? predictions[i * d + j + 3] : p.w;
                t.x = (k == 0) ? targets[i * d + j] : t.x;
                t.y = (k == 1) ? targets[i * d + j + 1] : t.y;
                t.z = (k == 2) ? targets[i * d + j + 2] : t.z;
                t.w = (k == 3) ? targets[i * d + j + 3] : t.w;
            }
        }

        // Compute partial sums
        dot += p.x * t.x + p.y * t.y + p.z * t.z + p.w * t.w;
        norm_pred += p.x * p.x + p.y * p.y + p.z * p.z + p.w * p.w;
        norm_targ += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
    }

    // Store partial sums in shared memory
    sh_dot[tid] = dot;
    sh_norm_pred[tid] = norm_pred;
    sh_norm_targ[tid] = norm_targ;
    __syncthreads();

    // Reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_dot[tid] += sh_dot[tid + s];
            sh_norm_pred[tid] += sh_norm_pred[tid + s];
            sh_norm_targ[tid] += sh_norm_targ[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 computes final loss
    if (tid == 0) {
        float final_dot = sh_dot[0];
        float final_norm_pred = sqrtf(sh_norm_pred[0]);
        float final_norm_targ = sqrtf(sh_norm_targ[0]);
        float denom = fmaxf(1e-8f, final_norm_pred) * fmaxf(1e-8f, final_norm_targ);
        output[i] = 1.0f - (final_dot / denom);
    }
}

extern "C" void solution(
    const float* predictions,
    const float* targets,
    float* output,
    size_t n,
    size_t d
) {
    int threadsPerBlock = 64; 
    int blocksPerGrid = n; 
    cosineSimilarityKernel<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, output, n, d);
}