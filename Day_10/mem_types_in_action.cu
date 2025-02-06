#include<iostream>
#include<cuda_runtime.h>

using namespace std;

// Constant memory (read-only for all threads)
__constant__ float const_data[4];

// Kernel function initialization 
__global__ void mem_types_demo(float *globalmem, float *output){
    // Thread idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared Memory
    __shared__ float shared_data[256];

    // Local Memory (pvt to each thread)
    float localVar = 0.0f; //0.0f is a float literal // Local variable

    // Register variable(Private to each thread)
    float regVar = 0.0f;

    // Conditional check
    if (idx < 256){
        // Loading mem from global to shared
        shared_data[threadIdx.x] = globalmem[idx];
        __syncthreads(); // Sync. threads in a block

        // For scaling: Using the constant memory...
        regVar = shared_data[threadIdx.x] * const_data[0];

        // Storing the intermediate result in the local memory
        __shared__ float localMem[256];
        localMem[threadIdx.x] = regVar + 1.0f;
        localVar = localMem[threadIdx.x];

        // Storing the final result in the global memory
        output[idx] = localVar;
    }
}

int main(){
    const int size = 256; // Num of elements in the array
    const int bytes = size * sizeof(float); // Size in bytes

    // Host arrays
    float h_input[size], h_output[size];

    // Initializing the input data on the host
    for (int i = 0; i < size; i++){
        h_input[i] = float(i);
    }

    // Device pointers
    float *d_input, *d_output;

    // Allocating memory on the device
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copying the input data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Constant memory initialization
    float scaleValue = 2.5f;
    cudaMemcpyToSymbol(const_data, &scaleValue, sizeof(float));

    // Kernel launch
    mem_types_demo<<<1, 256>>>(d_input, d_output);

    // Copying the output data from device to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Printing the output
    cout<<"Input (first 10 elements): ";
    for (int i = 0; i < 10; i++){
        cout<<h_input[i]<<" ";
    }
    cout<<endl;

    cout<<"Output (first 10 elements): ";
    for (int i = 0; i < 10; i++){
        cout<<h_output[i]<<" ";
    }
    cout<<endl;

    // Freeing up the memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}