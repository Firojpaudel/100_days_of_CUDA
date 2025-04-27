#include <cuda_runtime.h>
#include <math.h>

#define EPSILON 1e-10f

// Helper function for parallel reduction in shared memory
__device__ void reduceSum(float* sdata, int tid, int blockSize) {
    // Perform reduction in shared memory
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
}

__global__ void l2Normalize(
    const float* __restrict__ X, // Input tensor (B x D)
    float* __restrict__ Y,       // Output tensor (B x D)
    int B,                       // Number of rows
    int D                        // Number of columns
) {
    // Shared memory for partial sums of squares
    extern __shared__ float sdata[];

    int b = blockIdx.x; // One block per row
    int tid = threadIdx.x; // Thread index within block
    int blockSize = blockDim.x;

    if (b >= B) return; // Out-of-bounds check

    // Step 1: Compute partial sum of squares for this row
    float sum = 0.0f;
    for (int d = tid; d < D; d += blockSize) {
        float x = X[b * D + d];
        sum += x * x;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Step 2: Parallel reduction to compute total sum of squares
    reduceSum(sdata, tid, blockSize);

    // Step 3: Compute L2 norm (root sum of squares + epsilon)
    float l2_norm = 0.0f;
    if (tid == 0) {
        l2_norm = sqrtf(sdata[0]) + EPSILON;
        // Optionally store l2_norm if needed elsewhere
    }
    __syncthreads();

    // Step 4: Normalize each element
    for (int d = tid; d < D; d += blockSize) {
        Y[b * D + d] = X[b * D + d] / l2_norm;
    }
}

extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    if (B <= 0 || D <= 0) return;

    // Choose block size (tune based on GPU and D)
    int threadsPerBlock = 256;
    if (D < 256) threadsPerBlock = 128; // Adjust for small D
    int blocksPerGrid = B; // One block per row

    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // Launch kernel
    l2Normalize<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(X, Y, B, D);
}