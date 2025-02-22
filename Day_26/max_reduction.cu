#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

__global__ void maxReduction(int* input, int* output, int n) {
    // Allocating shared memory for the block
    extern __shared__ int sharedData[];

    // Getting thread and global indices
    int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Loading data into shared memory (handling out-of-bound indices)
    sharedData[tid] = (idx < n) ? input[idx] : INT_MIN;
    __syncthreads();

    // Performing parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }

    // Storing the final result of this block
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

int main() {
    int N;
    
    // Asking for the number of inputs first
    cout << "Enter the number of integers: ";
    cin >> N;

    if (N <= 0) {
        cerr << "Invalid input size." << endl;
        return 1;
    }

    vector<int> h_input(N);

    // Asking for the elements
    cout << "Enter " << N << " integers: ";
    for (int i = 0; i < N; ++i) {
        cin >> h_input[i];
    }

    int h_output;

    // Allocating device memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

    // Copying data from host to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Launching kernel with one block and N threads
    maxReduction<<<1, N, N * sizeof(int)>>>(d_input, d_output, N);

    // Copying the result back to host
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Printing the maximum value
    cout << "Max is " << h_output << endl;

    // Freeing allocated device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}