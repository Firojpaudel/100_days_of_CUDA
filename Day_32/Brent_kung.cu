#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// Defining SECTION_SIZE (shared memory size per block)
#define SECTION_SIZE 16

// Brent-Kung Scan Kernel
__global__ void Brent_Kung_scan_kernel(float *X, float *Y, unsigned int N)
{
    // Allocating shared memory
    __shared__ float XY[SECTION_SIZE];

    // Calculating thread's global index
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Loading input into shared memory
    if (i < N)
        XY[threadIdx.x] = X[i];
    else
        XY[threadIdx.x] = 0; // Padding for out-of-bound threads
    if (i + blockDim.x < N)
        XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    else
        XY[threadIdx.x + blockDim.x] = 0; // Padding

    __syncthreads();

    // Performing Reduction Tree Phase
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < SECTION_SIZE)
            XY[index] += XY[index - stride];
    }

    // Performing Reverse Tree Phase
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2)
    {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE)
            XY[index + stride] += XY[index];
    }

    __syncthreads();

    // Writing results back to global memory
    if (i < N)
        Y[i] = XY[threadIdx.x];
    if (i + blockDim.x < N)
        Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

int main()
{
    const unsigned int N = SECTION_SIZE; // Array size capped to 16 elements
    const unsigned int THREADS_PER_BLOCK = SECTION_SIZE / 2;

    // Allocating host memory
    float h_X[N], h_Y[N];

    // Asking user for input values
    cout << "Enter " << N << " values for the input array:" << endl;
    for (unsigned int i = 0; i < N; i++)
    {
        cin >> h_X[i];
    }

    // Allocating device memory
    float *d_X, *d_Y;
    cudaMalloc(&d_X, N * sizeof(float));
    cudaMalloc(&d_Y, N * sizeof(float));

    // Copying input data from host to device
    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculating grid and block dimensions
    unsigned int numBlocks = (N + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);

    // Measuring kernel execution time using chrono
    auto start_time = chrono::high_resolution_clock::now();

    // Launching the Brent-Kung scan kernel
    Brent_Kung_scan_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_X, d_Y, N);

    cudaDeviceSynchronize(); // Waiting for GPU to finish

    auto end_time = chrono::high_resolution_clock::now();

    // Copying results back to host
    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculating execution time in milliseconds
    auto duration_us = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    cout << "\nKernel Execution Time: " << duration_us << " micro_secs" << endl;

    // Displaying results
    cout << "\nInclusive Scan Result:" << endl;
    for (unsigned int i = 0; i < N; i++)
    {
        cout << h_Y[i] << " ";
    }
    cout << endl;

    // Freeing device and host memory
    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}