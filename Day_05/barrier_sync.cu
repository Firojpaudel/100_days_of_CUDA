#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void correctKernel(int *data) {
    int idx = threadIdx.x;

    // Phase 1: Each thread increments its own value.
    data[idx] += 1;

    // Ensuring all threads have completed Phase 1.
    __syncthreads();

    // Phase 2: Each thread doubles the value of its own.
    data[idx] *= 2;
}

__global__ void incorrectKernel(int *data) {
    int idx = threadIdx.x;

    // Phase 1: Each thread increments its own value.
    data[idx] += 1;

    // The condition that would create a deadlock:
    // Only threads with idx < 5 call __syncthreads().
    if (idx < 5) {
        __syncthreads();
    }

    // Phase 2: Each thread doubles the value of its own.
    data[idx] *= 2;
}

int main() {
    const int size = 10;
    int h_data[size];
    int *d_data;

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the correct kernel
    correctKernel<<<1, size>>>(d_data);
    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Results from the correct kernel function:" << endl;
    for (int i = 0; i < size; i++) {
        cout << h_data[i] << " ";
    }
    cout << endl;

    // Reinitialize input data
    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the incorrect kernel
    cout << "Launching the incorrect kernel (this may hang due to deadlock)..." << endl;
    incorrectKernel<<<1, size>>>(d_data);
    
    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cout << "Error in incorrectKernel: " << cudaGetErrorString(err) << endl;
    } else {
        cout << "Incorrect kernel executed (unexpectedly) without error." << endl;
        cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
        cout << "Results from the incorrect kernel function:" << endl;
        for (int i = 0; i < size; i++) {
            cout << h_data[i] << " ";
        }
        cout << endl;
    }

    // Free device memory
    cudaFree(d_data);

    return 0;
}