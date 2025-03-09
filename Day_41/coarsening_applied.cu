#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

using namespace std;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t error = call;                                                \
        if (error != cudaSuccess) {                                              \
            printf("CUDA error %04d: %s file: %s line: %d\n", error,             \
                   cudaGetErrorString(error), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

void printArray(int *arr, int n, const char *label) {
    cout << label << ": ";
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

__device__ int extractBits(int key, int bitOffset, int numBits) {
    return (key >> bitOffset) & ((1 << numBits) - 1);
}

__global__ void radix_sort_coarsened(int *input, int *output, int numKeys, int bitOffset, int numBits) {
    extern __shared__ int histogram[];
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int globalTid = blockIdx.x * blockDim.x + tid;
    
    for (int i = tid; i < (1 << numBits); i += blockSize) {
        histogram[i] = 0;
    }
    __syncthreads();

    int localPos[16];
    int localCount[16];
    int count = 0;
    
    for (int i = globalTid; i < numKeys; i += gridDim.x * blockDim.x) {
        int key = input[i];
        int bucket = extractBits(key, bitOffset, numBits);
        atomicAdd(&histogram[bucket], 1);
        localPos[count] = i;
        localCount[count] = bucket;
        count++;
    }
    __syncthreads();

    int total = 0;
    if (tid == 0) {
        for (int i = 0; i < (1 << numBits); i++) {
            int temp = histogram[i];
            histogram[i] = total;
            total += temp;
        }
    }
    __syncthreads();

    for (int i = 0; i < count; i++) {
        int bucket = localCount[i];
        int pos = atomicAdd(&histogram[bucket], 1);
        output[pos] = input[localPos[i]];
    }
}

int main() {
    const int numKeys = 16;
    const int numBitsPerIteration = 2;
    const int totalBits = 4;

    int h_input[numKeys] = {5, 14, 3, 5, 14, 8, 14, 14, 8, 14, 0, 9, 0, 8, 3, 14};
    int h_output[numKeys];

    printArray(h_input, numKeys, "Original Array");

    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(int) * numKeys));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int) * numKeys));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(int) * numKeys, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (numKeys + blockSize - 1) / blockSize;
    // Adjusted shared memory size since we only need histogram now
    const int sharedMemSize = (1 << numBitsPerIteration) * sizeof(int);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float totalTime = 0;

    int *d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, sizeof(int) * numKeys));

    for (int bitOffset = 0; bitOffset < totalBits; bitOffset += numBitsPerIteration) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(int) * numKeys));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        radix_sort_coarsened<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, numKeys, bitOffset, numBitsPerIteration);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime += milliseconds;
        
        CUDA_CHECK(cudaDeviceSynchronize());
        swap(d_input, d_output);
    }

    CUDA_CHECK(cudaMemcpy(h_output, d_input, sizeof(int) * numKeys, cudaMemcpyDeviceToHost));

    printArray(h_output, numKeys, "Sorted Array");
    
    cout << "Total kernel execution time: " << totalTime << " ms" << endl;
    cout << "Average time per iteration: " << totalTime / (totalBits / numBitsPerIteration) << " ms" << endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_temp));

    return 0;
}