#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// Defining constants and configuration
#define OUT_TILE_DIM 32
#define IN_TILE_DIM (OUT_TILE_DIM + 2)
#define N 256

// Stencil coefficients in constant memory
__constant__ float c0 = 6.0f;
__constant__ float c1 = -1.0f;
__constant__ float c2 = -1.0f;
__constant__ float c3 = -1.0f;
__constant__ float c4 = -1.0f;
__constant__ float c5 = -1.0f;
__constant__ float c6 = -1.0f;

__global__ void stencil_kernel(float* in, float* out, unsigned int n) {
    // Calculating thread indices with halo
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    // Register storage for temporal data
    float inPrev;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inCurr, inNext;

    // Loading initial data
    if (iStart - 1 >= 0 && iStart - 1 < n && j >= 0 && j < n && k >= 0 && k < n) {
        inPrev = in[(iStart - 1) * n * n + j * n + k];
    }

    if (iStart >= 0 && iStart < n && j >= 0 && j < n && k >= 0 && k < n) {
        inCurr = in[iStart * n * n + j * n + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }

    // Main computation loop
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        // Prefetch next element
        if (i + 1 >= 0 && i + 1 < n && j >= 0 && j < n && k >= 0 && k < n) {
            inNext = in[(i + 1) * n * n + j * n + k];
        }

        __syncthreads();

        // Performing stencil computation
        if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && 
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i * n * n + j * n + k] = 
                    c0 * inCurr +
                    c1 * inCurr_s[threadIdx.y][threadIdx.x - 1] +
                    c2 * inCurr_s[threadIdx.y][threadIdx.x + 1] +
                    c3 * inCurr_s[threadIdx.y + 1][threadIdx.x] +
                    c4 * inCurr_s[threadIdx.y - 1][threadIdx.x] +
                    c5 * inPrev +
                    c6 * inNext;
            }
        }

        __syncthreads();

        // Shifting data for next iteration
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}

void printWelcome() {
    cout << "=======================================================================\n"
         << "      3D STENCIL GPU ACCELERATOR: OPTIMIZED WITH REGISTER TILING       \n"
         << "=======================================================================\n"
         << " Grid Size: " << N << "x" << N << "x" << N << "\n"
         << " Tile Size: " << OUT_TILE_DIM << "x" << OUT_TILE_DIM << "\n"
         << " Shared Memory/Block: " 
         << IN_TILE_DIM*IN_TILE_DIM*sizeof(float)/1024 << " KB\n"
         << "=======================================================================\n";
}

int main() {
    printWelcome();
    
    float *h_in, *h_out, *d_in, *d_out;
    size_t size = N*N*N*sizeof(float);
    
    // Allocating memory
    h_in = new float[N*N*N];
    h_out = new float[N*N*N];
    fill_n(h_in, N*N*N, 1.0f);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Configuring kernel launch
    dim3 threads(IN_TILE_DIM, IN_TILE_DIM);
    dim3 blocks(
        (N + OUT_TILE_DIM - 1)/OUT_TILE_DIM,
        (N + OUT_TILE_DIM - 1)/OUT_TILE_DIM,
        (N + OUT_TILE_DIM - 1)/OUT_TILE_DIM
    );

    // Timing execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    stencil_kernel<<<blocks, threads>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculating performance
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cout << "\nExecution Summary:\n"
         << " Grid Dimensions: " << blocks.x << "x" << blocks.y << "x" << blocks.z << "\n"
         << " Block Dimensions: " << threads.x << "x" << threads.y << "\n"
         << " Total Time: " << ms << " ms\n"
         << " Throughput: " << (2*N*N*N*sizeof(float)) / (ms/1000) / 1e9 << " GB/s\n"
         << "=======================================================================\n";

    // Cleaning up
    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}
