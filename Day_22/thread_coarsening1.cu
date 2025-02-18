#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// Defining constants and configuration
#define OUT_TILE_DIM 32
#define IN_TILE_DIM (OUT_TILE_DIM + 2)
#define N 256

// Defining stencil coefficients (3D Laplace operator)
__constant__ float c0 = 6.0f;
__constant__ float c1 = -1.0f;
__constant__ float c2 = -1.0f;
__constant__ float c3 = -1.0f;
__constant__ float c4 = -1.0f;
__constant__ float c5 = -1.0f;
__constant__ float c6 = -1.0f;

__global__ void stencil_kernel(float* in, float* out) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

    // Checking boundaries and loading shared memory
    if (iStart-1 >= 0 && iStart-1 < N && j >=0 && j<N && k>=0 && k<N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart-1)*N*N + j*N + k];
    }

    if (iStart >=0 && iStart<N && j>=0 && j<N && k>=0 && k<N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N + k];
    }

    for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if(i+1 >=0 && i+1<N && j>=0 && j<N && k>=0 && k<N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i+1)*N*N + j*N + k];
        }
    }

    __syncthreads();

    // Performing stencil computation
    if(iStart >=1 && iStart<N-1 && j>=1 && j<N-1 && k>=1 && k<N-1) {
        if(threadIdx.y >=1 && threadIdx.y < IN_TILE_DIM-1 &&
           threadIdx.x >=1 && threadIdx.x < IN_TILE_DIM-1) {
            out[iStart*N*N + j*N + k] = 
                c0 * inCurr_s[threadIdx.y][threadIdx.x] +
                c1 * inCurr_s[threadIdx.y][threadIdx.x-1] +
                c2 * inCurr_s[threadIdx.y][threadIdx.x+1] +
                c3 * inCurr_s[threadIdx.y+1][threadIdx.x] +
                c4 * inCurr_s[threadIdx.y-1][threadIdx.x] +
                c5 * inPrev_s[threadIdx.y][threadIdx.x] +
                c6 * inNext_s[threadIdx.y][threadIdx.x];
        }
    }

    __syncthreads();
    inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
    inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
}

void printWelcome() {
    cout << "========================================\n"
         << "      3D STENCIL GPU ACCELERATOR       \n"
         << "========================================\n"
         << " Grid Size: " << N << "x" << N << "x" << N << "\n"
         << " Tile Size: " << OUT_TILE_DIM << "x" << OUT_TILE_DIM << "\n"
         << " Shared Memory/Block: " 
         << 3*IN_TILE_DIM*IN_TILE_DIM*sizeof(float)/1024 << " KB\n"
         << "========================================\n";
}

int main() {
    printWelcome();
    
    float *h_in, *h_out, *d_in, *d_out;
    size_t size = N*N*N*sizeof(float);
    
    // Allocating host memory
    h_in = new float[N*N*N];
    h_out = new float[N*N*N];
    
    // Initializing input
    fill_n(h_in, N*N*N, 1.0f);
    
    // Allocating device memory
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    
    // Copying data to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    // Configuring kernel
    dim3 threads(IN_TILE_DIM, IN_TILE_DIM);
    dim3 blocks((N + OUT_TILE_DIM -1)/OUT_TILE_DIM,
                (N + OUT_TILE_DIM -1)/OUT_TILE_DIM,
                (N + OUT_TILE_DIM -1)/OUT_TILE_DIM);
    
    // Creating CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Executing kernel with timing
    cudaEventRecord(start);
    stencil_kernel<<<blocks, threads>>>(d_in, d_out);
    cudaEventRecord(stop);
    
    // Waiting for kernel to finish
    cudaEventSynchronize(stop);
    
    // Calculating elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copying result back
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    // Printing performance info
    cout << "\nExecution Summary:\n"
         << " Grid Dimensions: " << blocks.x << "x" << blocks.y << "x" << blocks.z << "\n"
         << " Block Dimensions: " << threads.x << "x" << threads.y << "\n"
         << " Total Time: " << milliseconds << " ms\n"
         << " Throughput: " << (2*N*N*N*sizeof(float)) / (milliseconds/1000) / 1e9 << " GB/s\n"
         << "========================================\n";
    
    // Cleaning up
    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
