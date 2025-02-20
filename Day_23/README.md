## Summary of Day 23:

> **Well in day 23; I reinstalled the windows and Visual Studio kinda messed up my CUDA installation. But solved it after installing the previous version of Visual Studio (2019 Community Edition). However, I learnt a bit.* 

>**Completion of Chapter 8*

#### **Exercise from Chapter 8**

1. Consider a 3D stencil computation on a grid of size $120 \times 120 \times 120$,including boundary cells.
    1. What is the number of output grid points that is computed during each stencil sweep?
    2. For the basic kernel code below, what is the number of thread blocks that are needed, assuming a block size of $8\times8\times8$?
        ```cpp
        __global__ void stencil_kernel(float* in, float* out, unsigned int N) {
            unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
            unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
                out[i * N * N + j * N + k] = 
                    C0 * in[i * N * N + j * N + k]
                    + C1 * in[i * N * N + j * N + (k - 1)]
                    + C2 * in[i * N * N + j * N + (k + 1)]
                    + C3 * in[i * N * N + (j - 1) * N + k]
                    + C4 * in[i * N * N + (j + 1) * N + k]
                    + C5 * in[(i - 1) * N * N + j * N + k]
                    + C6 * in[(i + 1) * N * N + j * N + k];
            }
        }
        ```
    3.  For the kernel with shared memory tiling shown below, what is the number of thread blocks that are needed, assuming a block size of $8\times8\times8$?
        ```cpp
        __global__ void stencil_kernel(float* in, float* out, unsigned int N) {
            int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
            int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
            int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

            __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

            if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
                in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
            }

            __syncthreads();

            if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
                if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 &&
                    threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
                    threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                    
                    out[i * N * N + j * N + k] = 
                        C0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                        + C1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1]
                        + C2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1]
                        + C3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x]
                        + C4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                        + C5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x]
                        + C6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
                }
            }
        }
        ```

    ***Solution***:
    
Okay , so for ***1.1: Number of Output Grid Points per Stencil Sweep***

In a 3D Stencil computation, boundary cells are **typically excluded from the output calculation since their values are either fixed or require special handling**. 

For a grid size of $120 \times 120 \times 120$, the boundaries occupy the outermost layer in each dimension. This leaves an interior region of $(120-2)^3= 118^3$

_Calculating:_
$$118 \times 118 \times 118 = 1643032$$

Thus, $1,643,032$ output grid points are computed during each stencil sweep.
    
Now, coming to ***1.2: Number of Thread Blocks for the Basic Kernel***

The kernel uses a block size of $8\times8\times8$ threads. **Each thread computes one output point in the interior region**. To cover all 118 valid indices along each dimension:

$$\text{Blocks per dimension} = \lceil \frac{118}{8} \rceil= 15$$

Since the computation is in 3D, the total number of thread blocks is: $15^3 = 3,375$

Next, time for ***1.3 Number of Thread Blocks for the Shared Memory Kernel***

The **shared memory kernel loads a halo region around each output tile**. 

Let $\text{OUT TILE DIM}$ denote the output tile size per block, and $\text{IN TILE DIM} = \text{OUT TILE DIM} + 2$ account for the halo. 

Given the block size of $8\times8\times8$, we derive:

$$\text{OUT TILE DIM}= 8-2 = 6$$

Each block computes: $6 \times 6 \times 6$ output points. 

To cover $118$ points per dimension:

$$\text{Blocks Per Dimension} = \lceil \frac{118}{6} \rceil = 20$$

Therefore, 

$$\text{Total blocks required} = 20^3 = 8,000$$

---
<div align="center">
    <b>
        End of Day_23🫡
    </b>
</div>
