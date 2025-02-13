#include <cuda_runtime.h>
#include <iostream>
using namespace std;
// CUDA kernel for 2D convolution.
// N: input image, F: filter (kernel), P: output image,
__global__ void convolution2D(const float *N, const float *F, float *P, int width, int height, int filter_radius)
{
    // Compute output pixel coordinates (row and column) using block and thread indices
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    if (outRow < height && outCol < width)
    {
        float Pvalue = 0.0f;
        // Compute filter dimensions
        int filterDim = 2 * filter_radius + 1;
        // Iterate over the filter window
        for (int i = -filter_radius; i <= filter_radius; i++)
        {
            for (int j = -filter_radius; j <= filter_radius; j++)
            {
                int curRow = outRow + i;
                int curCol = outCol + j;
                // Check boundary conditions (ghost cells)
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                {
                    int filterRow = i + filter_radius;
                    int filterCol = j + filter_radius;
                    Pvalue += N[curRow * width + curCol] * F[filterRow * filterDim + filterCol];
                }
            }
        }
        // Store the result in the output image.
        P[outRow * width + outCol] = Pvalue;
    }
}
int main()
{
    // Define image dimensions and filter parameters.
    const int width = 16;
    const int height = 16;
    const int filter_radius = 1; // radius = 1 => filter size = 3x3
    const int filterDim = 2 * filter_radius + 1;
    // Allocate and initialize the host input image.
    float *h_N = new float[width * height];
    for (int i = 0; i < width * height; i++)
    {
        // For simplicity, fill the input with 1.0f.
        h_N[i] = 1.0f;
    }

    // Allocate and initialize the host filter.
    float *h_F = new float[filterDim * filterDim];
    // Create an averaging filter.
    for (int i = 0; i < filterDim * filterDim; i++)
    {
        h_F[i] = 1.0f / (filterDim * filterDim);
    }

    // Allocate host memory for the output image.
    float *h_P = new float[width * height];

    // Allocate device memory.
    float *d_N, *d_F, *d_P;
    cudaMalloc((void **)&d_N, width * height * sizeof(float));
    cudaMalloc((void **)&d_F, filterDim * filterDim * sizeof(float));
    cudaMalloc((void **)&d_P, width * height * sizeof(float));

    // Copy host memory to device.
    cudaMemcpy(d_N, h_N, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, filterDim * filterDim * sizeof(float), cudaMemcpyHostToDevice);

    // Set-up the execution configuration—arrange threads in a 2D grid.
    // Here we use a block of 16x16 threads (adjustable) and compute grid size accordingly.
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    convolution2D<<<gridSize, blockSize>>>(d_N, d_F, d_P, width, height, filter_radius);

    // Wait for the GPU to finish before accessing on host.
    cudaDeviceSynchronize();

    // Copy the result back to host.
    cudaMemcpy(h_P, d_P, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the output result.
    cout << "Convolution result:" << endl;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            cout << h_P[i * width + j] << " ";
        }
        cout << endl;
    }

    // Clean up device and host memory.
    cudaFree(d_N);
    cudaFree(d_F);
    cudaFree(d_P);
    delete[] h_N;
    delete[] h_F;
    delete[] h_P;

    return 0;
}