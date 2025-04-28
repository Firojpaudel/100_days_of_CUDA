#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void symMatMul(
    const float* __restrict__ A,    // Input matrix A (N x N, symmetric)
    const float* __restrict__ B,    // Input matrix B (N x N, symmetric)
    float* __restrict__ C,          // Output matrix C (N x N)
    int N                           // Matrix dimension
) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH]; // Tile of A
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH]; // Tile of B

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty; // Row index in C
    int col = blockIdx.x * TILE_WIDTH + tx; // Column index in C

    float sum = 0.0f;

    // Iterate over tiles along K dimension
    for (int k = 0; k < N; k += TILE_WIDTH) {
        // Load A: sA[ty][tx] = A[row, k + tx]
        if (row < N && (k + tx) < N) {
            sA[ty][tx] = A[row * N + (k + tx)];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load B: sB[ty][tx] = B[k + ty, col]
        if (col < N && (k + ty) < N) {
            sB[ty][tx] = B[(k + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum for this tile
        for (int i = 0; i < TILE_WIDTH && (k + i) < N; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }

    // Write to C if within bounds
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void solution(
    const float* A,
    const float* B,
    float* C,
    size_t N
) {
    if (N == 0) return;

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    symMatMul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, (int)N);
}