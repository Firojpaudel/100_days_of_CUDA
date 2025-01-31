#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__ void ex1 (int *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int j = 0; j < N; j++) {
            a[idx * N + j] = idx;
        }
    }
}

int main() {
    int width = 10;
    int size = width * width * sizeof(int);
    int *d_a, *h_a;

    // Allocate memory on the host
    h_a = (int*)malloc(size);

    // Allocate memory on the device
    cudaMalloc((void**)&d_a, size);

    dim3 threadsPerBlock(width);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x);
   
    // Kernel call
    ex1<<<numBlocks, threadsPerBlock>>>(d_a, width);
    
    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
    
    // Print section
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            cout << h_a[i * width + j] << " ";
        }
        cout << endl;
    }
    
    cudaFree(d_a);
    return 0;
}