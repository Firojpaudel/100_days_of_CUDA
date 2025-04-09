#include <cuda_runtime.h>

//kernel code:
__global__ void mat_scalar(const float* __restrict__ input_matrix, const float scalar, float* __restrict__ output_matrix, size_t n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n){
        int idx = row * n + col;
        output_matrix[idx] = scalar * input_matrix[idx];
    }
}

// Note: input_matrix, output_matrix are all device pointers to float32 arrays
extern "C" void solution(const float* input_matrix, const float scalar, float* output_matrix, size_t n) {    
    const int BLOCKSIZE_X = 64;
    const int BLOCKSIZE_Y = 16;

    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blockPerGrid(
        (n+ BLOCKSIZE_X -1 ) / BLOCKSIZE_X,
        (n+ BLOCKSIZE_Y -1 )/ BLOCKSIZE_Y
    );

    mat_scalar<<<blockPerGrid, threadsPerBlock>>>(input_matrix, scalar, output_matrix, n);

    cudaDeviceSynchronize();
}
