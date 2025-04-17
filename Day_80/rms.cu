#include <cuda_runtime.h>

__global__ void rmsNormKernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    size_t b,
    size_t n
) {
    size_t i = blockIdx.y; // Sample index
    extern __shared__ float sdata[];

    if (i >= b) return;

    // Step 1: Compute sum of squares for sample i
    float sum = 0.0f;
    for (size_t j = threadIdx.x * 4; j < n; j += blockDim.x * 4) {
        if (j + 3 < n) {
            float4 val = *reinterpret_cast<const float4*>(&x[i * n + j]);
            sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
        } else {
            if (j + 0 < n) sum += x[i * n + j + 0] * x[i * n + j + 0];
            if (j + 1 < n) sum += x[i * n + j + 1] * x[i * n + j + 1];
            if (j + 2 < n) sum += x[i * n + j + 2] * x[i * n + j + 2];
            if (j + 3 < n) sum += x[i * n + j + 3] * x[i * n + j + 3];
        }
    }

    // Reduce sum within block
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Compute RMS for sample i
    float rms = 0.0f;
    if (threadIdx.x == 0) {
        float mean = sdata[0] / static_cast<float>(n);
        rms = sqrtf(mean + 0.00001f);
    }
    __syncthreads();

    // Step 2: Normalize
    for (size_t j = threadIdx.x * 4; j < n; j += blockDim.x * 4) {
        if (j + 3 < n) {
            float4 val = *reinterpret_cast<const float4*>(&x[i * n + j]);
            float4 result;
            result.x = val.x / rms;
            result.y = val.y / rms;
            result.z = val.z / rms;
            result.w = val.w / rms;
            *reinterpret_cast<float4*>(&y[i * n + j]) = result;
        } else {
            if (j + 0 < n) y[i * n + j + 0] = x[i * n + j + 0] / rms;
            if (j + 1 < n) y[i * n + j + 1] = x[i * n + j + 1] / rms;
            if (j + 2 < n) y[i * n + j + 2] = x[i * n + j + 2] / rms;
            if (j + 3 < n) y[i * n + j + 3] = x[i * n + j + 3] / rms;
        }
    }
}

extern "C" void solution(
    const float* x,
    float* y,
    size_t b,
    size_t n
) {
    dim3 threadsPerBlock(256, 1);
    dim3 blocksPerGrid(1, b); // One block per sample
    size_t sharedMemSize = threadsPerBlock.x * sizeof(float);
    rmsNormKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(x, y, b, n);
}