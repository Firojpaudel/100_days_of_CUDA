#include<iostream>

#include<cuda_runtime.h>
#include<chrono>

using namespace std;

// Kernel without any dynamic parallelism
// Under recursion: lets calculate a factorial (easy representation :) )
__global__ void factrorial_kernel (unsigned long long* result, int n){
    unsigned long long temp = 1;
    for (int i =1; i<= n; i++){
        temp *= i;
    }
    *result = temp;
}

//Host code
unsigned long long factorial_host(int n){
    if (n <= 1){
        // Will resolve the negative problem in int call 
        return 1;
    }
    else{
        return n* factorial_host(n-1); 
    }
}

int main (){
    int n;
    cout<<"Enter the number you want factorial of (Enter smaller numbers larger numbers will overflow):";
    cin >> n;

    if (n<0){
        cout <<"The number must be non-negative one"<<endl;
        return 1;
    }

    // if (n> 20){
    //     cout << "Number to large limiting to 20" <<endl;
    //     n= 20;
    // }

    //Device Memory Allocation 
    unsigned long long* d_result;
    cudaMalloc(&d_result, sizeof(unsigned long long));

    //GPU timing
    auto start_gpu = chrono::high_resolution_clock::now();

    //launching the kernel — with 1 block and 1 thread
    factrorial_kernel<<<1,1>>>(d_result, n);
    cudaDeviceSynchronize();

    //GPU timing end
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli>gpu_time = end_gpu- start_gpu;

    //Back to the host
    unsigned long long h_result;
    cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    //Host time calculation 
    auto start_host = chrono::high_resolution_clock::now();
    unsigned long long host_result = factorial_host(n);
    auto end_host = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> host_time = end_host - start_host;

    //Displayingg
    cout << "GPU Result: " <<h_result <<endl;
    cout << "GPU Time of Execution (no dynamic parallelism):"<<gpu_time.count()<<"ms"<<endl;
    cout << "Host Result: " << host_result << endl;
    cout << "Host Time: " << host_time.count() << " ms" << endl;

    // Cleanup timeeee
    cudaFree(d_result);

    // Error Check 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        cout<<"Ayo! you got the CUDA Error:" << cudaGetErrorString(err) <<endl;
    }

    return 0;
}