## Summary of Day 12:

 > *First explanation of the code  of yesterday's "the tiled matrix multiplication" code.

#### Code Explanation for [Tiled Matrix Multiplication](../Day_11/tiled_mat_mul.cu):

1. **Program Structure:** 

    The code consists of:
    - CUDA kernel for tiled matrix multiplication (`tiledMatrixMulKernel`)
    - CPU matrix multiplication with OpenMP parallelization (`cpuMatMul`)
    - Main driver handling input/output, memory management, and benchmarking

2. Core Algorithm:

    - GPU Kernel (Tiled Approach):

    ```cpp
    __global__ void tiledMatrixMulKernel(float *M, float *N, float *P, int Width) {
        __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

        int tx = threadIdx.x, ty = threadIdx.y;
        int bx = blockIdx.x, by = blockIdx.y;

        int Row = by * TILE_WIDTH + ty;
        int Col = bx * TILE_WIDTH + tx;
        float Pvalue = 0;

        for (int ph = 0; ph < (Width + TILE_WIDTH - 1)/TILE_WIDTH; ++ph) {
            int mIdx = Row * Width + ph * TILE_WIDTH + tx;
            int nIdx = (ph * TILE_WIDTH + ty) * Width + Col;
            
            Mds[ty][tx] = (Row < Width && ph*TILE_WIDTH + tx < Width) ? M[mIdx] : 0;
            Nds[ty][tx] = (Col < Width && ph*TILE_WIDTH + ty < Width) ? N[nIdx] : 0;

            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k)
                Pvalue += Mds[ty][k] * Nds[k][tx];
            __syncthreads();
        }

        if (Row < Width && Col < Width)
            P[Row * Width + Col] = Pvalue;
    }
    ```
> **Key GPU Optimizations:**
> - Shared memory tiles $(32\times32)$ reduce global memory accesses
> - Thread coarsening through loop unrolling
Boundary condition handling for non-divisible matrix sizes
> - Bank conflict-free shared memory access patterns

- **CPU Implementation:**

    ```cpp
    void cpuMatMul(float *A, float *B, float *C, int N) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                for (int k = 0; k < N; ++k)
                    C[i*N + j] += A[i*N + k] * B[k*N + j];
    }
    ```
> - Uses OpenMP for multi-core parallelization
> - Naive $O(n^3)$ algorithm without cache optimization

3. **Execution Workflow**
    1. **Matrix Initialization:**
        - Creates $N×N$ matrices with random float values $(0.0-1.0)$
        - Allocates host and (conditional) device memory
    2. **Device Selection:**
        ```cpp
        bool useGPU = N > THRESHOLD;  // THRESHOLD = 256
        ```
        - Uses CPU for small matrices ($<256$ elements), GPU for larger ones
    3. **Benchmarking:**
        - CPU timing with `<chrono>`
        - GPU timing with CUDA events
        - Proper CUDA stream synchronization
    4. **Validation:**
        - Compares CPU and GPU results
        - Calculates maximum absolute error

4. Performance Considerations
    1. **Memory Hierarchy Utilization:**
        - Global → Shared → Register data movement
        - Coalesced global memory accesses
    2. **Thread Organization:**
        - 2D thread blocks $(32×32 = 1024 \space \text{threads/block})$
        - Grid size calculation: `⌈N/32⌉ × ⌈N/32⌉`
    3. **Compute Characteristics:**
        - *Operational intensity:* $O(n)$ operations per byte
        - *Theoretical peak performance:*
            ```math
            \text{GFLOPS} = \frac{2 \times N^3}{\text{time taken}} \times 10^{-9}
            ```
    <br><br>
    > This implementation demonstrates effective heterogeneous computing by:
    >- Leveraging CPU strengths for small problems
    >- Utilizing GPU massive parallelism for large datasets
    >- Maintaining numerical consistency between devices
    >- Providing automated performance optimization based on problem size


#### Impact of memory Usage on Occupancy:

1. **Occupancy Fundamentals**

    Occupancy refers to the ratio of active warps to maximum possible warps per SM. _Key factors affecting occupancy_:
    - Register usage: Each thread's register consumption
    - Shared memory: Per-block shared memory allocation
    - Thread block size: Warp occupancy patterns
    
    For matrix multiplication (TILE_WIDTH=32):
    ```cpp
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // 32x32 = 1024 elements
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; // 32x32 = 1024 elements
    ```
    - Total shared memory per block: $2 \times 32² \times 4\text{B} = 8\text{KB}$
    - Threads per block: $1024$ $(32 \times 32)$
    - Shared memory per thread: $8\text{KB} / 1024 = 8\text{B}$


> ***Resource Limitations Example (A100 GPU)***
>
>   |Resource|Capacity|
>    |---|---|
>   |Shared Memory/SM	|164 KB|
>   |Max Threads/SM	|2048|
>   |Registers/SM	|65536|

2. **Dynamic Shared Memory Allocation**

    Modified kernel implementation code segment:
    ```cpp
    __global__ void matrixMulKernel(float* M, float* N, float* P, int Width, size_t Mds_sz, size_t Nds_sz) {
        extern __shared__ char shared_mem[];
        float* Mds = (float*)shared_mem;
        float* Nds = (float*)&shared_mem[Mds_sz];
        
        // Rest of kernel code using 1D access:
        // Mds[ty * tile_width + tx] instead of Mds[ty][tx]
    }
    ```
    > For Full Implementation, [Click Here](./Day_12_updated_code.cu)

3. **Performance Optimization Strategy**

    1. Device Query:
    ```cpp
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t max_shared_per_block = prop.sharedMemPerBlock;
    ```

    2. Adaptive Tile Sizing:
    ```cpp
    size_t optimal_tile = sqrt(prop.sharedMemPerBlock / (2 * sizeof(float)));
    optimal_tile = min(optimal_tile, prop.maxThreadsDim[0]);
    ```

    3. Occupancy Calculator:
    ```cpp
    int max_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, 
                                                matrixMulKernel, 
                                                threads_per_block, 
                                                shared_mem_size);
    ```

> ***Performance Comparison:***
> |**Approach**	|**Flexibility**	|**Portability**	|**Occupancy Control**	|**Code Complexity**|
> |---|---|---|---|---|
>|Static	|Low	|Poor	|Compile-time	|Simple|
>Dynamic	|High	|Excellent	|Runtime	|Moderate