#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

#define TILE_SIZE 16  // Tile size for shared memory

// Naive Matrix Multiplication (Global Memory Only)
__global__ void matMulNaive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];  // Unoptimized access
        }
        C[row * N + col] = sum;
    }
}

// Optimized Matrix Multiplication (Corner-Turning + Shared Memory)
__global__ void matMulOptimized(float *A, float *B, float *C, int N) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;

    for (int t = 0; t < (N / TILE_SIZE); t++) {
        // Load tiles into shared memory
        Asub[ty][tx] = A[row * N + (t * TILE_SIZE + tx)];
        Bsub[tx][ty] = B[(t * TILE_SIZE + ty) * N + col]; // Corner-turning optimization

        __syncthreads();  // Sync threads

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++)
            sum += Asub[ty][k] * Bsub[k][tx];

        __syncthreads();  // Sync before next tile
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Function to initialize matrices with random values
void initMatrix(float *matrix, int N) {
    for (int i = 0; i < N * N; i++)
        matrix[i] = rand() % 10;
}

// Main function
int main() {
    int N;
    cout << "Enter the size of the matrix (N x N) _N must be multiple of_"<<TILE_SIZE<<":";
    cin >> N;

    if (N % TILE_SIZE != 0) {
        cout << "Matrix size N must be a multiple of " << TILE_SIZE << endl;
        return -1;
    }

    size_t size = N * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C1 = (float*)malloc(size);  // Naive
    float *h_C2 = (float*)malloc(size);  // Optimized

    // Initialize matrices
    srand(time(NULL));
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C1, *d_C2;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C1, size);
    cudaMalloc(&d_C2, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure grid/block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(N / TILE_SIZE, N / TILE_SIZE);

    // CUDA event timing
    cudaEvent_t start, stop;
    float naiveTime, optimizedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run Naive Multiplication
    cudaEventRecord(start);
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C1, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&naiveTime, start, stop);

    // Run Optimized Multiplication
    cudaEventRecord(start);
    matMulOptimized<<<grid, block>>>(d_A, d_B, d_C2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&optimizedTime, start, stop);

    // Copy results back
    cudaMemcpy(h_C1, d_C1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, size, cudaMemcpyDeviceToHost);

    // Print timing results
    cout << "Naive Multiplication Time: " << naiveTime << " ms" << endl;
    cout << "Optimized Multiplication Time: " << optimizedTime << " ms" << endl;

    // Free memory
    free(h_A); free(h_B); free(h_C1); free(h_C2);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C1); cudaFree(d_C2);
    
    return 0;
}
