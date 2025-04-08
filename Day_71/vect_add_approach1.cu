#include <cuda_runtime.h>

//kernel -- approach loop unrolling --> Increasing this to process 4 elements per thread
__global__ void vect_addn(const float* a, const float* b, float* c, int n){
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int stride = blockDim.x;

     if (idx + 3 * stride < n) {
        c[idx] = a[idx] + b[idx];
        c[idx + stride] = a[idx + stride] + b[idx + stride];
        c[idx + 2 * stride] = a[idx + 2 * stride] + b[idx + 2 * stride];
        c[idx + 3 * stride] = a[idx + 3 * stride] + b[idx + 3 * stride];
    }
}

// Note: d_input1, d_input2, d_output are all device pointers to float32 arrays
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
    // Launch the kernel
    vect_addn<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_output, n);

    cudaDeviceSynchronize();
}