#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define NUM_THREADS 8  // Number of threads in the block

__global__ void compute_waiting_time(float *exec_times, float *waiting_percentage) {
    __shared__ float max_time;
    __shared__ float total_waiting_time;

    int tid = threadIdx.x;

    // Step 1: Find max execution time using reduction
    if (tid == 0) max_time = exec_times[0]; // Initialize shared max_time
    __syncthreads();

    atomicMax((int*)&max_time, __float_as_int(exec_times[tid]));
    __syncthreads();

    // Step 2: Compute waiting time per thread
    float waiting_time = max_time - exec_times[tid];

    // Step 3: Sum up total waiting time using reduction
    if (tid == 0) total_waiting_time = 0; // Initialize shared total_waiting_time
    __syncthreads();

    atomicAdd(&total_waiting_time, waiting_time);
    __syncthreads();

    // Step 4: Compute percentage of time spent waiting
    if (tid == 0) {
        float total_exec_time = NUM_THREADS * max_time;
        *waiting_percentage = (total_waiting_time / total_exec_time) * 100.0f;
    }
}

int main() {
    float h_exec_times[NUM_THREADS] = {2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9};
    float *d_exec_times, *d_waiting_percentage;
    float h_waiting_percentage;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_exec_times, NUM_THREADS * sizeof(float));
    cudaMalloc((void**)&d_waiting_percentage, sizeof(float));

    // Copy execution times to GPU
    cudaMemcpy(d_exec_times, h_exec_times, NUM_THREADS * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and NUM_THREADS threads
    compute_waiting_time<<<1, NUM_THREADS>>>(d_exec_times, d_waiting_percentage);

    // Copy result back to CPU
    cudaMemcpy(&h_waiting_percentage, d_waiting_percentage, sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Percentage of time spent waiting: %.2f%%\n", h_waiting_percentage);

    // Free memory
    cudaFree(d_exec_times);
    cudaFree(d_waiting_percentage);

    return 0;
}
