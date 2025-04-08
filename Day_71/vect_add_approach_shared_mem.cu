#include <cuda_runtime.h>

//kernel -- approach:: Shared memory
__global__ void vect_addn(const float* a, const float* b, float* c, int n) {
    extern __shared__ float s_data[]; // Shared memory for a and b
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        s_data[tid] = a[idx];                // Loading from a
        s_data[tid + blockDim.x] = b[idx];   // Loadng from b
    }
    __syncthreads();

    if (idx < n) {
        c[idx] = s_data[tid] + s_data[tid + blockDim.x];
    }
}
// Note: d_input1, d_input2, d_output are all device pointers to float32 arrays
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = 2 * threadsPerBlock * sizeof(float); 

    vect_addn<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input1, d_input2, d_output, n);
    cudaDeviceSynchronize();
}