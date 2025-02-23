#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// Simple Sum Reduction Kernel
__global__ void SimpleSumReductionKernel(float* input, float* output, int N) {
    unsigned int i = threadIdx.x;

    for (unsigned int stride = 1; stride < N; stride *= 2) {
        int index = 2 * stride * i;
        if (index + stride < N) {  // Bounds checking
            input[index] += input[index + stride];
        }
        __syncthreads();
    }

    if (i == 0) {
        *output = input[0];
    }
}

// Convergent Sum Reduction Kernel
__global__ void ConvergentSumReductionKernel(float* input, float* output, int N) {
    unsigned int i = threadIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

// CPU Reference Sum Function
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

    if (N & (N - 1)) {  // Ensuring N is power of 2
        cout << "Error: N must be a power of 2." << endl;
        return 1;
    }

    size_t size = N * sizeof(float);

    // Allocating host memory
    float* h_input = (float*)malloc(size);
    float h_output1, h_output2, cpu_sum;

    // Getting array values from user
    cout << "Enter " << N << " float values:" << endl;
    for (int i = 0; i < N; i++) {
        cin >> h_input[i];
    }

    // Allocating device memory
    float *d_input1, *d_input2, *d_output1, *d_output2;
    cudaMalloc((void**)&d_input1, size);
    cudaMalloc((void**)&d_input2, size);
    cudaMalloc((void**)&d_output1, sizeof(float));
    cudaMalloc((void**)&d_output2, sizeof(float));

    // Copying input data to device
    cudaMemcpy(d_input1, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input, size, cudaMemcpyHostToDevice);

    // CUDA event timers
    cudaEvent_t start, stop;
    float time1, time2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timing Simple Sum Reduction Kernel
    cudaEventRecord(start);
    SimpleSumReductionKernel<<<1, N/2>>>(d_input1, d_output1, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);

    // Timing Convergent Sum Reduction Kernel
    cudaEventRecord(start);
    ConvergentSumReductionKernel<<<1, N/2>>>(d_input2, d_output2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time2, start, stop);

    // Copying results back to host
    cudaMemcpy(&h_output1, d_output1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_output2, d_output2, sizeof(float), cudaMemcpyDeviceToHost);

 
    cpu_sum = computeSumCPU(h_input, N);

    // Printing results
    cout << "\nResults:" << endl;
    cout << "CPU Sum: " << cpu_sum << endl;
    cout << "Simple Sum Reduction (Kernel 1): " << h_output1 << " (Time: " << time1 << " ms)" << endl;
    cout << "Convergent Sum Reduction (Kernel 2): " << h_output2 << " (Time: " << time2 << " ms)" << endl;

    // Checking correctness
    if (fabs(h_output1 - cpu_sum) < 1e-5 && fabs(h_output2 - cpu_sum) < 1e-5) {
        cout << "Both kernels match CPU result!" << endl;
    } else {
        cout << "Mismatch detected! Debugging needed." << endl;
    }

    // Freeing memory
    free(h_input);
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}