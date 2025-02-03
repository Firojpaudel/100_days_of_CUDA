#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__ void loops_warp_divergence_example(){
    int tid = threadIdx.x; 

    int iterations = (tid % 5)+ 1; // Number of iterations for each thread
    int value = 0;

    // Loop divergence
    for (int i = 0; i < 10; i++){
        value += tid;
    }

    printf("Thread %2d: Iterations = %d, Sum = %d\n", tid, iterations, value);
}

int main(){
    loops_warp_divergence_example<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}