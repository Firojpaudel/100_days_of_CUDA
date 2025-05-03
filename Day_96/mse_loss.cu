#include <cuda_runtime.h>
#include <cstdio>


__global__ void mseKernel(const float* __restrict__ predictions, const float* __restrict__ targets, size_t numElements, float* __restrict__ sum) {
    extern __shared__ float sdata[];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;

    float sq_diff = 0.0f;
    if (idx < numElements) {
        float diff = predictions[idx] - targets[idx];
        sq_diff = diff * diff;
    }


    sdata[tid] = sq_diff;
    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }


    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t* shape, size_t ndim) {

    size_t* hostShape = new size_t[ndim];
    cudaMemcpy(hostShape, shape, ndim * sizeof(size_t), cudaMemcpyDeviceToHost);

    size_t numElements = 1;
    for (size_t i = 0; i < ndim; i++) {
        numElements *= hostShape[i];
    }
    delete[] hostShape;

    float init = 0.0f;
    cudaMemcpy(output, &init, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    mseKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(predictions, targets, numElements, output);
    cudaDeviceSynchronize();

    float hostSum = 0.0f;
    cudaMemcpy(&hostSum, output, sizeof(float), cudaMemcpyDeviceToHost);
    float mse = hostSum / numElements;

    cudaMemcpy(output, &mse, sizeof(float), cudaMemcpyHostToDevice);
}