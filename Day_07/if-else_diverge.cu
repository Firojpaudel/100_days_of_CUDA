#include<iostream>
using namespace std;

__global__ void if_else_divergence_example(){
    int tid = threadIdx.x; 

    int value = 0;

    // if-else divergence
    if (tid % 2 == 0){
        value = tid * 2; // For even thread ids
    } else {
        value = tid * 3; // For odd thread ids
    }

    printf("Thread %d: Value = %d\n", tid, value);
}

int main(){
    if_else_divergence_example<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}