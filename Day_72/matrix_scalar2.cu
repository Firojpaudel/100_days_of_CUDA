#include <cuda_runtime.h>

//kernel code:
__global__ void mat_scalar(const float* __restrict__ input_matrix, const float scalar, float* __restrict__ output_matrix, size_t n){
    
    //Shared Mem...
    const int BLOCKSIZE = 32;
    __shared__ float tile[BLOCKSIZE][BLOCKSIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //Loading into shared memory
    if (row < n && col < n){
        int idx = row * n + col;
        tile[threadIdx.y][threadIdx.x] = input_matrix[idx];
    }else{
        tile[threadIdx.y][threadIdx.x] = 0.0f; //padding: outofbounds
    }

    __syncthreads();

    //Finally into the output matrix
    if (row < n && col < n){
        int idx = row * n + col;
        output_matrix[idx] = tile[threadIdx.y][threadIdx.x] * scalar;
    }
}

// Note: input_matrix, output_matrix are all device pointers to float32 arrays
extern "C" void solution(const float* input_matrix, const float scalar, float* output_matrix, size_t n) {    
    const int BLOCKSIZE = 32;

    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blockPerGrid(
        (n+ BLOCKSIZE -1 ) / BLOCKSIZE,
        (n+ BLOCKSIZE -1 )/ BLOCKSIZE
    );

    mat_scalar<<<blockPerGrid, threadsPerBlock>>>(input_matrix, scalar, output_matrix, n);

    cudaDeviceSynchronize();
}