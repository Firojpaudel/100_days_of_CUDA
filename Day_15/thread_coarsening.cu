#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

using namespace std;

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

// CPU Matrix Multiplication for comparison
void matrixMultiplyCPU(float *M, float *N, float *P, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += M[i * width + k] * N[k * width + j];
            }
            P[i * width + j] = sum;
        }
    }
}

// GPU Kernel with coarse-grained parallelism
__global__ void matrixMulCoarsened(float *M, float *N, float *P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH * COARSE_FACTOR];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int colStart = blockIdx.x * (TILE_WIDTH * COARSE_FACTOR) + tx;

    float Pvalues[COARSE_FACTOR];

    for (int m = 0; m < width / TILE_WIDTH; ++m) {
        Mds[ty][tx] = M[row * width + (m * TILE_WIDTH + tx)];

        for (int i = 0; i < COARSE_FACTOR; ++i) {
            int colIdx = colStart + i * TILE_WIDTH;
            Nds[ty][tx + i * TILE_WIDTH] = N[(m * TILE_WIDTH + ty) * width + colIdx];
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            for (int i = 0; i < COARSE_FACTOR; ++i) {
                Pvalues[i] += Mds[ty][k] * Nds[k][tx + i * TILE_WIDTH];
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < COARSE_FACTOR; ++i) {
        int colIdx = colStart + i * TILE_WIDTH;
        if (row < width && colIdx < width) {
            P[row * width + colIdx] = Pvalues[i];
        }
    }
}

// Host function to run and compare GPU execution
void matrixMultiplyHost(float *M, float *N, float *P, int width) {
    int size = width * width * sizeof(float);
    float *d_M, *d_N, *d_P;

    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(width / (TILE_WIDTH * COARSE_FACTOR), width / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulCoarsened<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
    cudaEventRecord(stop);

    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Compute GFLOPS
    float gflops = (2.0f * width * width * width) / (milliseconds * 1.0e6);
    cout << "GPU Execution Time: " << milliseconds << " ms" << endl;
    cout << "GPU Throughput: " << gflops << " GFLOPS" << endl;

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main() {
    int width = 512;
    float *M = new float[width * width];
    float *N = new float[width * width];
    float *P_GPU = new float[width * width]; // Product from GPU
    float *P_CPU = new float[width * width]; // Product from CPU

    srand(time(0)); // Seed for random number generation

    // Initialize matrices
    for (int i = 0; i < width * width; ++i) {
        M[i] = static_cast<float>(rand()) / RAND_MAX; // RAND_MAX is the maximum value returned by rand()
        N[i] = static_cast<float>(rand()) / RAND_MAX; 
    }

    cout << "Running GPU matrix multiplication..." << endl;
    matrixMultiplyHost(M, N, P_GPU, width);

    // Measure CPU Execution Time
    clock_t cpu_start = clock();
    matrixMultiplyCPU(M, N, P_CPU, width);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    cout << "CPU Execution Time: " << cpu_time << " ms" << endl;

    // Print a small part of the matrices for verification
    cout << "\nSample Output Comparison (First 5x5 block):\n";
    cout << "GPU Result:\n";
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            cout << P_GPU[i * width + j] << "\t";
        }
        cout << endl;
    }

    cout << "\nCPU Result:\n";
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            cout << P_CPU[i * width + j] << "\t";
        }
        cout << endl;
    }

    // Cleanup
    delete[] M;
    delete[] N;
    delete[] P_GPU;
    delete[] P_CPU;

    return 0;
}