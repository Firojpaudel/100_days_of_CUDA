#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel for Sum Reduction (Fixed Version)
__global__ void SimpleSumReductionKernel(float* input, float* output, int N) {
    unsigned int i = threadIdx.x;

    // Performing reduction iteratively
    for (unsigned int stride = 1; stride < N; stride *= 2) {
        int index = 2 * stride * i;
        if (index + stride < N) {  // Ensuring within bounds
            input[index] += input[index + stride];
        }
        __syncthreads();  // Ensuring all threads finish before next step
    }

    // Storing the final sum
    if (i == 0) {
        *output = input[0];
    }
}

// CPU function to compute sum
float computeSumCPU(float* arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    int N;

    // Getting array size from user
    cout << "Enter the number of elements (power of 2): ";
    cin >> N;

    if (N & (N - 1)) {  // Checking if N is power of 2
        cout << "Error: N must be a power of 2." << endl;
        return 1;
    }

    size_t size = N * sizeof(float);

    // Allocating host memory
    float* h_input = (float*)malloc(size);
    float h_output, cpu_sum;

    // Getting array values from user
    cout << "Enter " << N << " float values:" << endl;
    for (int i = 0; i < N; i++) {
        cin >> h_input[i];
    }

    // Allocating device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, sizeof(float));

    // Copying data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launching kernel with one block and N/2 threads
    SimpleSumReductionKernel<<<1, N/2>>>(d_input, d_output, N);

    // Copying result back to host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Computing sum using CPU for verification
    cpu_sum = computeSumCPU(h_input, N);

    // Printing results
    cout << "CUDA Sum: " << h_output << endl;
    cout << "CPU Sum: " << cpu_sum << endl;

    // Checking correctness
    if (fabs(h_output - cpu_sum) < 1e-5) {
        cout << "Results match! :)" << endl;
    } else {
        cout << "Mismatch detected! Debugging needed. :(" << endl;
    }

    // Freeing memory
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
