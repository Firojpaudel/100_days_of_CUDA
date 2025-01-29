#include<iostream>
#include<cuda_runtime.h>
using namespace std;

__global__ void vectadd_kernel(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

void vectadd(float *A, float *B, float *C, int n){ // Host function
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;  // Device/GPU pointers 

    /*Part 1: Allocate the memory for A, B and C*/
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    /*Part 2: Transfering variables to the device before Kernel Call*/
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    /*Part 2: Kernel Call*/
    vectadd_kernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);  // Num_blocks = ceil(n/256.0), Num_threads = 256

    /*Part 3: Device to Host Memory Transfer*/
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    /*Part 4: Free the device memory*/
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

int main() {  // Host Code
    int n = 1000;
    size_t size = n * sizeof(float); 
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize vectors
    for(int i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Perform vector addition
    vectadd(h_A, h_B, h_C, n);

    // Display first and last elements of the vector
    cout << "C[0] = " << h_C[0] << endl;
    cout << "C[" << n-1 << "] = " << h_C[n-1] << endl;

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}