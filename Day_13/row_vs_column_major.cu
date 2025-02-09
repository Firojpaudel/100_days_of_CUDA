#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Kernel for Row Major Order
__global__ void rowMajor(float *matrix, float *output, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int index = row * N + col;
        output[index] = matrix[index] * 2.0f;
    }
}

// Kernel for Column Major Order
__global__ void columnMajor(float *matrix, float *output, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int index = col * N + row;
        output[index] = matrix[index] * 2.0f;
    }
}

int main() {
    int N;
    
    // Taking user input for matrix size
    cout << "Enter the matrix size (N x N): ";
    cin >> N;

    if (N <= 0 || N > 5000) { // Setting an upper limit for safety
        cerr << "Invalid size! Please enter a value between 1 and 5000." << endl;
        return -1;
    }

    size_t size = N * N * sizeof(float);

    // Allocate memory on the host dynamically
    float *h_matrix = new float[N * N];
    float *h_output = new float[N * N];

    // Initialize the matrix with some values
    for (int i = 0; i < N * N; i++) {
        h_matrix[i] = static_cast<float>(i);
    }

    // Allocate memory on the device
    float *d_matrix, *d_output;
    cudaMalloc((void**)&d_matrix, size);
    cudaMalloc((void**)&d_output, size);

    // Copy matrix from host to device
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Execution time measurement
    cudaEvent_t startRow, stopRow, startCol, stopCol;
    float timeRow, timeCol;

    cudaEventCreate(&startRow);
    cudaEventCreate(&stopRow);
    cudaEventCreate(&startCol);
    cudaEventCreate(&stopCol);

    // Running Row Major Kernel
    cudaEventRecord(startRow);
    rowMajor<<<gridSize, blockSize>>>(d_matrix, d_output, N);
    cudaEventRecord(stopRow);
    cudaEventSynchronize(stopRow);
    cudaEventElapsedTime(&timeRow, startRow, stopRow);

    // Running Column Major Kernel
    cudaEventRecord(startCol);
    columnMajor<<<gridSize, blockSize>>>(d_matrix, d_output, N);
    cudaEventRecord(stopCol);
    cudaEventSynchronize(stopCol);
    cudaEventElapsedTime(&timeCol, startCol, stopCol);

    // Copying the output from device to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Printing execution times
    cout << "Row Major Execution Time: " << timeRow << " ms" << endl;
    cout << "Column Major Execution Time: " << timeCol << " ms" << endl;

    // Freeing memory
    delete[] h_matrix;
    delete[] h_output;
    cudaFree(d_matrix);
    cudaFree(d_output);

    // Destroy CUDA events
    cudaEventDestroy(startRow);
    cudaEventDestroy(stopRow);
    cudaEventDestroy(startCol);
    cudaEventDestroy(stopCol);

    return 0;
}
