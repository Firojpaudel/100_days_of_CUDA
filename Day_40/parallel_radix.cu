#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <algorithm> // For std::sort

/**
 * Robust CUDA error-checking macro
 * Checks the CUDA call result and provides detailed error information if the call fails.
 * This is crucial for debugging CUDA applications as errors can be cryptic.
 */
#define CHECK_CUDA(call) {                                                  \
    cudaError_t err = call;                                                 \
    if(err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA Error at %s:%d - %s (err_num=%d): %s\n",      \
                __FILE__, __LINE__, #call, err, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

/**
 * Parallel Radix Sort Kernel for Binary (0/1) Keys
 * 
 * This kernel implements a work-efficient parallel radix sort for an array of binary keys.
 * The algorithm sorts values in-place using a single pass of least-significant-bit (LSB) radix sort,
 * which is sufficient for binary (0/1) values.
 *
 * Algorithm Overview:
 * ------------------
 * 1. Each thread loads one input element and determines if it's a 0 or 1
 * 2. We compute an exclusive prefix sum (scan) of the "1" flags across all elements
 * 3. This prefix sum tells us how many 1's occur before each position
 * 4. Using this information, we can calculate the final position for each element:
 *    - For elements with value 0: position = (thread_id - prefix_sum)
 *    - For elements with value 1: position = (total_zeros + prefix_sum)
 * 5. We use shared memory buckets to reorder the elements
 * 6. Finally, we write the sorted result back to global memory in a coalesced pattern
 *
 * Assumptions:
 * -----------
 * - Input size (n) is a power of 2
 * - All elements in the input array are either 0 or 1
 * - Input array fits in a single thread block (typically ≤ 1024 elements)
 *
 * Shared Memory Usage:
 * -------------------
 * - s_scan  [0...n-1]: Stores prefix sum of the "1" flags
 * - s_zeros [0...n-1]: Temporary bucket for all "0" elements
 * - s_ones  [0...n-1]: Temporary bucket for all "1" elements
 *
 * @param d_input   Input array in device memory (containing only 0's and 1's)
 * @param d_output  Output array in device memory (will contain the sorted result)
 * @param n         Number of elements in the input array (must be power of 2)
 */
__global__ void radix_sort_kernel(const int *d_input, int *d_output, int n)
{
    // Allocate shared memory for our algorithm:
    extern __shared__ int s_mem[];
    int *s_scan  = s_mem;        // For prefix sum calculation (n elements)
    int *s_zeros = s_mem + n;    // Bucket for "0" elements (n elements)
    int *s_ones  = s_mem + 2*n;  // Bucket for "1" elements (n elements)

    int tid = threadIdx.x;  // Thread ID within the block

    // Step 1: Load input data and determine if element is 0 or 1
    int key = 0;    // The value we're sorting
    int flag = 0;   // 1 if key is 1, 0 if key is 0 
    
    if(tid < n) {
        key = d_input[tid];              // Load element from global memory
        flag = (key == 1) ? 1 : 0;       // Set flag based on key value
        s_scan[tid] = flag;              // Store flag in shared memory for scan
    }
    __syncthreads();  // Ensure all threads have loaded their data

    // Step 2: Perform exclusive prefix sum (Blelloch scan) on the flags
    // This gives us the count of "1" values that come before each position
    
    // 2a: Upsweep (reduce) phase - build a sum tree from leaves to root
    for (int offset = 1; offset < n; offset *= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < n) {
            s_scan[index] += s_scan[index - offset];
        }
        __syncthreads();  // Synchronize before next iteration
    }

    // 2b: Downsweep phase - distribute sums back down the tree
    if(tid == 0) {
        s_scan[n - 1] = 0;  // Clear the last element (exclusive scan)
    }
    __syncthreads();
    
    for (int offset = n / 2; offset > 0; offset /= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if(index < n) {
            // Swap and accumulate pattern of downsweep
            int temp = s_scan[index - offset];
            s_scan[index - offset] = s_scan[index];
            s_scan[index] += temp;
        }
        __syncthreads();  // Synchronize before next iteration
    }
    // At this point, s_scan[i] contains the number of 1's before position i

    // Step 3: Compute total counts of 0's and 1's
    __shared__ int totalOnes, totalZeros;
    if(tid == n - 1) {
        // Last thread computes the totals:
        // - Total 1's = prefix sum at last position + flag at last position
        // - Total 0's = n - total 1's
        totalOnes = s_scan[n - 1] + flag;
        totalZeros = n - totalOnes;
    }
    __syncthreads();  // Ensure totals are visible to all threads

    // Step 4: Calculate destination indices and store elements in local buckets
    if(tid < n) {
        int prefix = s_scan[tid];  // Number of 1's before this position
        
        if(key == 0) {
            // For 0's: place at position (tid - prefix) in 0's bucket
            // This works because prefix = number of 0's pushed backward by 1's
            int dest = tid - prefix;
            s_zeros[dest] = key;
        } else {  // key == 1
            // For 1's: place at position (prefix) in 1's bucket
            // This works because prefix = position in the 1's section
            int dest = prefix;
            s_ones[dest] = key;
        }
    }
    __syncthreads();  // Ensure all elements are placed in buckets

    // Step 5: Copy data from local buckets to global memory in coalesced manner
    // First write all zeros, then all ones
    
    // 5a: Write zeros (from s_zeros to beginning of output array)
    for (int i = tid; i < totalZeros; i += blockDim.x) {
        d_output[i] = s_zeros[i];
    }
    
    // 5b: Write ones (from s_ones to after all zeros in output array)
    for (int i = tid; i < totalOnes; i += blockDim.x) {
        d_output[totalZeros + i] = s_ones[i];
    }
}

/**
 * Main function: Sets up test data, launches the CUDA kernel, and verifies results
 */
int main()
{
    // Initialize random seed
    srand(time(NULL));
    
    // Set problem size (must be power of 2 for this implementation)
    const int n = 16;
    size_t bytes = n * sizeof(int);

    // Allocate and initialize host arrays
    int h_input[n];
    int h_output[n];
    int h_sorted_reference[n];  // For verification purposes

    // Generate random binary data (0s and 1s only)
    std::cout << "=============================================" << std::endl;
    std::cout << "CUDA Radix Sort for Binary (0/1) Keys" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "\nInput array: ";
    int zeros_count = 0;
    int ones_count = 0;
    
    for(int i = 0; i < n; i++) {
        h_input[i] = rand() % 2;
        std::cout << h_input[i] << " ";
        
        // Count zeros and ones for verification
        if(h_input[i] == 0) zeros_count++;
        else ones_count++;
    }
    std::cout << std::endl;
    
    // Create and sort the reference array correctly
    // For binary values, this is simply: all zeros followed by all ones
    for(int i = 0; i < n; i++) {
        h_sorted_reference[i] = (i < zeros_count) ? 0 : 1;
    }
    
    // Input analysis
    std::cout << "\nInput Analysis:" << std::endl;
    std::cout << "- Array size: " << n << " elements" << std::endl;
    std::cout << "- Zeros count: " << zeros_count << " (" << std::fixed << std::setprecision(1) 
              << (zeros_count * 100.0 / n) << "%)" << std::endl;
    std::cout << "- Ones count: " << ones_count << " (" << std::fixed << std::setprecision(1) 
              << (ones_count * 100.0 / n) << "%)" << std::endl;

    // Allocate device memory
    int *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Configure kernel execution parameters
    int threads = n;                            // One thread per element
    int blocks = 1;                             // Single block for this example
    int sharedMemSize = 3 * n * sizeof(int);    // For s_scan, s_zeros, and s_ones

    // Setup timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup kernel call (to prime the GPU)
    radix_sort_kernel<<<blocks, threads, sharedMemSize>>>(d_input, d_output, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Performance measurement: Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Launch the radix sort kernel
    radix_sort_kernel<<<blocks, threads, sharedMemSize>>>(d_input, d_output, n);

    // Record stop event and synchronize
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate and report kernel execution time
    float elapsedTime;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "- Kernel execution time: " << elapsedTime << " ms" << std::endl;
    std::cout << "- Throughput: " << (n / elapsedTime) << " elements/ms" << std::endl;
    
    // Print the sorted output
    std::cout << "\nSorted output: ";
    for(int i = 0; i < n; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Print the expected output (for comparison)
    std::cout << "Expected output: ";
    for(int i = 0; i < n; i++) {
        std::cout << h_sorted_reference[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify the results
    bool correct = true;
    for(int i = 0; i < n; i++) {
        if(h_output[i] != h_sorted_reference[i]) {
            correct = false;
            std::cout << "Error at index " << i << ": expected " 
                      << h_sorted_reference[i] << ", got " << h_output[i] << std::endl;
            break;
        }
    }
    
    if(correct) {
        std::cout << "\nVerification: PASSED :)" << std::endl;
    } else {
        std::cout << "\nVerification: FAILED :(" << std::endl;
    }
       
    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    std::cout << "\n=============================================" << std::endl;

    return 0;
}