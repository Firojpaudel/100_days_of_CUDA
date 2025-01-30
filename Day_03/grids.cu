/* A program demonstrating the use of CUDA grids and blocks */
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

dim3 dimGrid(32, 1, 1);  // 32 blocks in the grid
dim3 dimBlock(128, 1, 1);  // 128 threads in each block

__global__ void kernel(void){}

int main(void) {
    kernel<<<dimGrid, dimBlock>>>();
    cudaDeviceSynchronize();  // Ensure the kernel has finished executing before exiting
    int totalThreads = dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x * dimBlock.y * dimBlock.z;
    cout << "Total number of threads: " << totalThreads << endl;
    return 0;
}