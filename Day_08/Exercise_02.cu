#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__ void vector_add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main(){
    const int n = 2000; // vector length
    const int threadsPerBlock = 512;

    const int Blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    const int total_threads = Blocks * threadsPerBlock;

    printf("Number of blocks: %d\n", Blocks);
    printf("Number of threads per block: %d\n", threadsPerBlock);
    printf("Total number of threads: %d\n", total_threads);
}