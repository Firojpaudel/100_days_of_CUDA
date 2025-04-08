#include <cuda_runtime.h>

//kernel -- approach:: Shared memory + loop unrolling
__global__ void vect_addn(const float* a, const float* b, float* c, int n) {
    extern __shared__ float s_data[];
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x; // Process 4 elements per thread
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Load 4 elements from a and b into shared memory
    if (idx + 3 * stride < n) {
        s_data[tid] = a[idx];
        s_data[tid + stride] = a[idx + stride];
        s_data[tid + 2 * stride] = a[idx + 2 * stride];
        s_data[tid + 3 * stride] = a[idx + 3 * stride];

        s_data[tid + 4 * stride] = b[idx];
        s_data[tid + 5 * stride] = b[idx + stride];
        s_data[tid + 6 * stride] = b[idx + 2 * stride];
        s_data[tid + 7 * stride] = b[idx + 3 * stride];
    }
    __syncthreads();

    // Perform addition and write back 4 elements
    if (idx + 3 * stride < n) {
        c[idx] = s_data[tid] + s_data[tid + 4 * stride];
        c[idx + stride] = s_data[tid + stride] + s_data[tid + 5 * stride];
        c[idx + 2 * stride] = s_data[tid + 2 * stride] + s_data[tid + 6 * stride];
        c[idx + 3 * stride] = s_data[tid + 3 * stride] + s_data[tid + 7 * stride];
    }
}

// Note: d_input1, d_input2, d_output are all device pointers to float32 arrays
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
    size_t sharedMemSize = 8 * threadsPerBlock * sizeof(float); // 4x for a, 4x for b

    vect_addn<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input1, d_input2, d_output, n);
    cudaDeviceSynchronize();
}