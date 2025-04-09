#include <cuda_runtime.h>

//kernel - simple approach first
__global__ void mat_vect (const float* __restrict__ input_a, const float* __restrict__ input_b, float* __restrict__ output_c, size_t M, size_t K){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M){
        float sum = 0.0f;
        for (int k = 0; k < K; k++){
            sum += input_a[row * K + k] * input_b[k];
        }
        output_c[row] = sum;
    }
}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t M, size_t K) {    
    const int BLOCKSIZE = 1024;

    dim3 threadsPerBlock(BLOCKSIZE);
    dim3 blocksPerGrid((M + BLOCKSIZE - 1) / BLOCKSIZE);

    mat_vect<<<blocksPerGrid, threadsPerBlock>>>(input_a, input_b, output_c, M, K);
    cudaDeviceSynchronize();
}