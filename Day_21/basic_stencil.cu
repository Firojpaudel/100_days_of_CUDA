#include <iostream>
#include <cuda_runtime.h>
#include <chrono>  // For measuring operation time

using namespace std;

#define N 32  // Defining the size of the 3D grid
#define c0 0.5f
#define c1 0.1f
#define c2 0.1f
#define c3 0.1f
#define c4 0.1f
#define c5 0.05f
#define c6 0.05f

// Desigining Welcome Banner 
void display_welcome_banner() {
    cout << "*******************************************************\n";
    cout << "*         PARALLEL STENCIL KERNEL PROGRAM             *\n";
    cout << "*                 A BASIC ALGORITHM                   *\n";
    cout << "*******************************************************\n";
    cout << endl;
}

// Defining the CUDA kernel for applying the stencil operation
__global__ void stencil_kernel(float* in, float* out, unsigned int size) {
    // Calculating the 3D indices of the current thread
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensuring the indices are within valid range (ignoring boundary points)
    if (i >= 1 && i < size - 1 && j >= 1 && j < size - 1 && k >= 1 && k < size - 1) {
        // Applying the 3D stencil operation using neighboring elements
        out[i * size * size + j * size + k] = 
              c0 * in[i * size * size + j * size + k]
            + c1 * in[i * size * size + j * size + k - 1]
            + c2 * in[i * size * size + j * size + k + 1]
            + c3 * in[i * size * size + (j - 1) * size + k]
            + c4 * in[i * size * size + (j + 1) * size + k]
            + c5 * in[(i - 1) * size * size + j * size + k]
            + c6 * in[(i + 1) * size * size + j * size + k];
    }
}

int main() {
    //Displaying the welcome banner
    display_welcome_banner();
    
    // Allocating memory for the 3D array on host (CPU)
    size_t total_size = N * N * N; // Total number of elements
    float* h_in = (float*)malloc(total_size * sizeof(float));
    float* h_out = (float*)malloc(total_size * sizeof(float));

    // Initializing the input array with some values
    for (int i = 0; i < total_size; i++) {
        h_in[i] = static_cast<float>(i % 100) / 100.0f;  // Assigning values between [0 and 1]
    }

    // Allocating memory on the GPU
    float* d_in, * d_out;
    cudaMalloc((void**)&d_in, total_size * sizeof(float));
    cudaMalloc((void**)&d_out, total_size * sizeof(float));
    
    // Copying input data from host to device
    cudaMemcpy(d_in, h_in, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initializing the output array to zero on the device
    cudaMemset(d_out, 0, total_size * sizeof(float));

    // Defining thread block and grid dimensions
    dim3 blockSize(8, 8, 8); // Using an 8x8x8 block of threads
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (N + blockSize.y - 1) / blockSize.y, 
                  (N + blockSize.z - 1) / blockSize.z);

    // Measuring operation time using chrono library
    auto start = chrono::high_resolution_clock::now();

    // Launching the CUDA kernel for processing the 3D data
    stencil_kernel<<<gridSize, blockSize>>>(d_in, d_out, N);

    // Waiting for CUDA to finish execution
    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    
    // Calculating elapsed time in milliseconds
    chrono::duration<double, milli> elapsed_time = end - start;
    
    cout << "Stencil computation completed in " << elapsed_time.count() << " ms.\n";

    // Copying the result back to host memory
    cudaMemcpy(h_out, d_out, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Checking a few values from the output array to ensure correctness
    cout << "Checking some values from the output array:\n";
    
    for (int i = N*N*(N/2)+N/2*N+N/2; i < N*N*(N/2)+N/2*N+N/2+10; i++) {
        cout << "h_out[" << i << "] = " << h_out[i] << endl;
        /* Printing values from a central region of the grid to ensure meaningful results */
    }

    // Freeing GPU memory
    cudaFree(d_in);
    cudaFree(d_out);

    // Freeing host memory
    free(h_in);
    free(h_out);

    return 0;
}