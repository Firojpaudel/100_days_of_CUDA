#include <cuda_runtime.h>


__global__ void lowerTriangularMatMul(
    const float* A,    // Input matrix A
    const float* B,    // Input matrix B
    float* C,          // Output matrix C
    int N              // Matrix size (N x N)
) {
    // Compute row (i) and column (j) indices for this thread
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    // Check if indices are within bounds
    if (i >= N || j >= N) return;

    // Only compute for lower triangular part (i >= j)
    if (i >= j) {
        float sum = 0.0f;
        // Sum from k = j to k = i
        for (int k = j; k <= i; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    } else {
        // Upper triangular part is zero
        C[i * N + j] = 0.0f;
    }
}


// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N) {    
    if (N <= 0) return; // Handle empty matrices

    dim3 threadsPerBlock(32, 32);
    // Compute number of blocks needed in each dimension
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch the kernel
    lowerTriangularMatMul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    
    //And well this is optional ...
    cudaDeviceSynchronize();
}