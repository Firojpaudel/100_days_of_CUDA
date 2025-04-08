#include <cuda_runtime.h>
#include <iostream>

// Tanh function implementation
__device__ float manual_tanh(float x){
    float pos_exp = __expf(x);
    float neg_exp = __expf(-x);

    return ((pos_exp - neg_exp) / (pos_exp + neg_exp));
}

// Kernel for tanh appication
__global__ void tanh_kernel(const float* input, float* output, int rows, int columns){
    int m = blockIdx.y * blockDim.y + threadIdx.y; //row
    int n = blockIdx.x * blockDim.x + threadIdx.x; //column

    // Bounds Check 
    if (m < rows && n < columns){
        int indx = m * columns + n;

        output[indx] = manual_tanh(input[indx]);
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {    
    //blockDim 
    dim3 blockDim(32,32);

    //gridDim
    dim3 gridDim ((n+ blockDim.x + 1)/ blockDim.x, (m+ blockDim.y + 1)/ blockDim.y);

    tanh_kernel<<<gridDim, blockDim>>>(input, output, m, n);

    cudaDeviceSynchronize();
}
