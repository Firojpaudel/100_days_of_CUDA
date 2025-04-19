#include <cuda_runtime.h>

__global__ void reductionKernel(
    const float* __restrict__ x,
    float* __restrict__ partialSums,
    size_t size
) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;

    float sum = 0.0f;
    if (i + 3 < size) {
        float4 val = *reinterpret_cast<const float4*>(&x[i]);
        sum = val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    } else {
        sum += (i + 0 < size) ? x[i + 0] * x[i + 0] : 0.0f;
        sum += (i + 1 < size) ? x[i + 1] * x[i + 1] : 0.0f;
        sum += (i + 2 < size) ? x[i + 2] * x[i + 2] : 0.0f;
        sum += (i + 3 < size) ? x[i + 3] * x[i + 3] : 0.0f;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(partialSums, sdata[0]);
    }
}

__global__ void normalizeKernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    float sum, // Pass sum instead of norm
    size_t size
) {
    float norm = sqrtf(sum); // Compute square root on device

    size_t i = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;

    if (i + 3 < size) {
        float4 val = *reinterpret_cast<const float4*>(&x[i]);
        float4 result;
        result.x = val.x / norm;
        result.y = val.y / norm;
        result.z = val.z / norm;
        result.w = val.w / norm;
        *reinterpret_cast<float4*>(&y[i]) = result;
    } else {
        if (i + 0 < size) y[i + 0] = x[i + 0] / norm;
        if (i + 1 < size) y[i + 1] = x[i + 1] / norm;
        if (i + 2 < size) y[i + 2] = x[i + 2] / norm;
        if (i + 3 < size) y[i + 3] = x[i + 3] / norm;
    }
}

extern "C" void solution(
    const float* x,
    float* y,
    size_t size
) {
    // Allocate device memory for partial sums
    float* partialSums;
    cudaMalloc(&partialSums, sizeof(float));
    cudaMemset(partialSums, 0, sizeof(float));

    // Reduction kernel
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((size + threadsPerBlock.x * 4 - 1) / (threadsPerBlock.x * 4));
    size_t sharedMemSize = threadsPerBlock.x * sizeof(float);
    reductionKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(x, partialSums, size);

    // Copy sum to host
    float sum;
    cudaMemcpy(&sum, partialSums, sizeof(float), cudaMemcpyDeviceToHost);

    // Normalization kernel (pass sum instead of norm)
    normalizeKernel<<<blocksPerGrid, threadsPerBlock>>>(x, y, sum, size);

    // Free device memory
    cudaFree(partialSums);
}