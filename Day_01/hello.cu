#include<iostream>

/*The global qualifier indicates that the function is a kernel function 
that runs on the device and is called from the host.*/
__global__ void kernel(void){
}

int main (void){
    kernel<<<1,1>>>(); // The angle brackets denote the arguments we plan to pass to the runtime system.
    printf("Hello, World!");
    return 0;
}