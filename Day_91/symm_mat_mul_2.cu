#include <cuda_runtime.h>

#define TILE_B 32
#define TILE_M 32
#define TILE_K 16
#define THREADS_X 16
#define THREADS_Y 16
#define REG_TILE_B 2
#define REG_TILE_M 2

__global__ void symMatMul(
    const float* __restrict__ A,    // N x N, symmetric
    const float* __restrict__ B,    // N x N, symmetric
    float* __restrict__ C,          // N x N
    int N
) {
    __shared__ float sA[TILE_B][TILE_K]; // 32 x 16
    __shared__ float sB[TILE_K][TILE_M]; // 16 x 32

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int base_row = blockIdx.y * TILE_B; // Base row of C
    int base_col = blockIdx.x * TILE_M; // Base col of C
    int local_b = 2 * ty;               // Local row offset (0, 2, 4, ...)
    int local_m = 2 * tx;               // Local col offset (0, 2, 4, ...)

    // Register tile for 2x2 submatrix of C
    float acc[REG_TILE_B][REG_TILE_M] = {0.0f};

    for (int k = 0; k < N; k += TILE_K) {
        // Cooperative loading
        int idx = ty * THREADS_X + tx;
        for (int i = 0; i < 2; i++) {
            int flat_idx = idx + i * (THREADS_X * THREADS_Y);

            // Load A: sA[a_row][a_col] = A[base_row + a_row, k + a_col]
            int a_row = flat_idx / TILE_K;
            int a_col = flat_idx % TILE_K;
            int global_row = base_row + a_row;
            int global_col = k + a_col;
            sA[a_row][a_col] = (global_row < N && global_col < N) ? A[global_row * N + global_col] : 0.0f;

            // Load B: sB[b_row][b_col] = B[k + b_row, base_col + b_col]
            int b_row = flat_idx / TILE_M;
            int b_col = flat_idx % TILE_M;
            int global_k = k + b_row;
            int global_col_b = base_col + b_col;
            sB[b_row][b_col] = (global_k < N && global_col_b < N) ? B[global_k * N + global_col_b] : 0.0f;
        }
        __syncthreads();

        // Compute 2x2 submatrix
        #pragma unroll
        for (int i = 0; i < TILE_K; i++) {
            float a_reg[REG_TILE_B];
            a_reg[0] = sA[local_b + 0][i];
            a_reg[1] = sA[local_b + 1][i];
            float b_reg[REG_TILE_M];
            b_reg[0] = sB[i][local_m + 0];
            b_reg[1] = sB[i][local_m + 1];

            acc[0][0] += a_reg[0] * b_reg[0];
            acc[0][1] += a_reg[0] * b_reg[1];
            acc[1][0] += a_reg[1] * b_reg[0];
            acc[1][1] += a_reg[1] * b_reg[1];
        }
        __syncthreads();
    }

    // Write 2x2 submatrix to C
    #pragma unroll
    for (int i = 0; i < REG_TILE_B; i++) {
        int r = base_row + local_b + i;
        if (r >= N) continue;
        #pragma unroll
        for (int j = 0; j < REG_TILE_M; j++) {
            int c = base_col + local_m + j;
            if (c >= N) continue;
            C[r * N + c] = acc[i][j];
        }
    }
}

extern "C" void solution(
    const float* A,
    const float* B,
    float* C,
    size_t N
) {
    if (N == 0) return;

    dim3 threadsPerBlock(THREADS_X, THREADS_Y); // 16 x 16 = 256 threads
    dim3 blocksPerGrid((N + TILE_M - 1) / TILE_M, (N + TILE_B - 1) / TILE_B);

    symMatMul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, (int)N);
}