#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void sumReduction(int* input, int* output, int n) {
    extern __shared__ int sharedData[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Loading data into shared memory
    if (idx < n) {
        sharedData[tid] = input[idx];
    } else {
        sharedData[tid] = 0; // Padding for out-of-bound threads
    }
    __syncthreads();

    // Performing reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            sharedData[index] += sharedData[index + stride];
        }
        __syncthreads();
    }

    // Writing result of this block to output
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main() {
    int N;
    cout << "Enter the number of elements: ";
    cin >> N;

    int* h_input = new int[N];
    cout << "Enter the elements: ";
    for (int i = 0; i < N; ++i) {
        cin >> h_input[i];
    }

    int h_output;

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launching kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sumReduction<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_input, d_output, N);

    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Sum is " << h_output << endl;

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    return 0;
}
