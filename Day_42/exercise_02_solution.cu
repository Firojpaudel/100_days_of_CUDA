#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void radix_sort_multibit_iter(unsigned int *input, unsigned int *output,
                                         unsigned int *histogram, unsigned int N,
                                         unsigned int iter, unsigned int bits_per_iter)
{
    extern __shared__ unsigned int shared_mem[];
    const unsigned int radix = 1U << bits_per_iter; // Radix defined here, innit
    unsigned int *bin_counts = shared_mem;          // First part of shared memory for counts
    unsigned int *prefix_sums = &shared_mem[radix]; // Second part for prefix sums

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // Clear out shared memory for the histogram
    if (tid < radix)
    {
        bin_counts[tid] = 0;
    }
    __syncthreads();

    // Build the histogram
    if (i < N)
    {
        unsigned int key = input[i];
        unsigned int digit = (key >> iter) & (radix - 1);
        histogram[i] = digit;
        atomicAdd(&bin_counts[digit], 1);
    }
    __syncthreads();

    // Thread 0 handles the prefix sums
    if (tid == 0)
    {
        prefix_sums[0] = 0;
        for (unsigned int r = 1; r < radix; r++)
        {
            prefix_sums[r] = prefix_sums[r - 1] + bin_counts[r - 1];
        }
    }
    __syncthreads();

    // Scatter the elements to their spots
    if (i < N)
    {
        unsigned int digit = histogram[i];
        unsigned int count_before = 0;
        // Count how many same digits are before this one
        for (unsigned int j = 0; j < i; j++)
        {
            if (histogram[j] == digit)
                count_before++;
        }
        unsigned int dst = prefix_sums[digit] + count_before;
        output[dst] = input[i];
    }
}

void radix_sort_multibit(unsigned int *h_input, unsigned int N, unsigned int bits_per_iter)
{
    unsigned int *d_input, *d_output, *d_histogram;
    const unsigned int radix = 1U << bits_per_iter;

    cudaMalloc(&d_input, N * sizeof(unsigned int));
    cudaMalloc(&d_output, N * sizeof(unsigned int));
    cudaMalloc(&d_histogram, N * sizeof(unsigned int));

    cudaMemcpy(d_input, h_input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = 2 * radix * sizeof(unsigned int);

    for (unsigned int iter = 0; iter < 32; iter += bits_per_iter)
    {
        radix_sort_multibit_iter<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, d_histogram, N, iter, bits_per_iter);
        cudaDeviceSynchronize();
        // Swap input and output
        unsigned int *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    cudaMemcpy(h_input, d_input, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_histogram);
}

int main()
{
    const unsigned int N = 8;
    unsigned int h_input[N] = {23, 1, 45, 12, 67, 3, 89, 15};
    unsigned int bits_per_iter = 2; // Radix-4

    cout << "Before sorting:\n";
    for (int i = 0; i < N; i++)
        cout << h_input[i] << " ";
    cout << "\n";

    radix_sort_multibit(h_input, N, bits_per_iter);

    cout << "After sorting:\n";
    for (int i = 0; i < N; i++)
        cout << h_input[i] << " ";
    cout << "\n";

    return 0;
}