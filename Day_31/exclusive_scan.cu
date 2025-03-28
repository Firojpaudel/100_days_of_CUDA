#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define SECTION_SIZE 1024 // Number of elements per block (adjustable)

// Kogge-Stone Exclusive Scan Kernel
__global__ void Kogge_Stone_exclusive_scan(float* X, float* Y, int N) {
    __shared__ float XY[SECTION_SIZE]; // Shared memory for the block

    int i = blockIdx.x * blockDim.x + threadIdx.x; // Global index

    // Load input data into shared memory with a shift for exclusive scan
    if (threadIdx.x == 0) {
        XY[threadIdx.x] = 0.0f; // Identity value for addition
    } else if (i - 1 < N) {
        XY[threadIdx.x] = X[i - 1];
    } else {
        XY[threadIdx.x] = 0.0f; // Padding with identity value
    }
    __syncthreads();

    // Perform Kogge-Stone scan within each block
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (threadIdx.x >= stride) {
            temp = XY[threadIdx.x - stride];
        }
        __syncthreads(); // Ensure all threads have read their values
        XY[threadIdx.x] += temp;
        __syncthreads(); // Ensure all threads have updated their values
    }

    // Write results back to global memory
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }
}

int main() {
    const int N = 16;
    float h_X[N];
    float h_Y[N];

    float *d_X, *d_Y;

    // Taking user input for the array
    cout << "Enter " << N << " elements for the input array:\n";
    for (int i = 0; i < N; ++i) {
        cin >> h_X[i];
    }

    // Allocating device memory
    cudaMalloc((void**)&d_X, N * sizeof(float));
    cudaMalloc((void**)&d_Y, N * sizeof(float));

    // Copying input data to device
    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launching kernel
    int threadsPerBlock = min(N, SECTION_SIZE);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Start the timer
    auto start = high_resolution_clock::now();

    Kogge_Stone_exclusive_scan<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, N);

    // Ensure the kernel has finished before stopping the timer
    cudaDeviceSynchronize();

    // Stop the timer
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    // Copying results back to host
    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Printing results
    cout << "Input Array: ";
    for (int i = 0; i < N; ++i) {
        cout << h_X[i] << " ";
    }
    cout << "\n";

    cout << "Output Array (Exclusive Scan): ";
    for (int i = 0; i < N; ++i) {
        cout << h_Y[i] << " ";
    }
    cout << "\n";

    cout << "Execution time: " << duration.count() / 1000.0f << " ms" << endl;

    // Freeing device memory
    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}