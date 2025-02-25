## Summary of Day 29

> *Exercises from Chapter 10

***Exercises***:
1.  For the simple reduction kernel in code below, if the number of elements is $1024$ and the warp size is $32$, how many warps in the block will have divergence during the fifth iteration?

    ```cpp
    __global__ void SharedMemorySumReductionKernel(float* input, float* output) {
        __shared__ float input_s[BLOCK_DIM];
        unsigned int t = threadIdx.x;

        input_s[t] = input[t] + input[t + BLOCK_DIM];
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
            if (threadIdx.x < stride) {
                input_s[t] += input_s[t + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *output = input_s[0];
        }
    }
    ```

***Solution:***

For the given reduction kernel with 1024 elements and a warp size of 32, 1 warp will experience divergence during the fifth iteration. Here's the breakdown:

1. **Initial Setup:**
    - The kernel starts with `num_threads = 1024` (matching the number of elements) and iteratively halves the active threads.

2. **Iteration Analysis:**
    - By the fifth iteration, the number of active threads is reduced to:
    ```math 
    \frac{1024}{2^5} = 32 \space \text{threads}
    ```
    - These $32$ threads fit into exactly **1 warp** (since $32 \div 32 = 1$).

3. **Divergence Mechanism:**
    - In CUDA, divergence occurs when threads in the same warp follow different execution paths. During the fifth iteration:

        - Only threads with `threadIdx.x < 32` execute the reduction logic.

        - This condition splits the single active warp between threads executing the reduction and those idling, creating divergence within that warp.

Thus, **1 warp** contains a mix of active and inactive threads during the fifth iteration, leading to divergence.

---

2. For the improved reduction kernel in code below, if the number of elements is $1024$ and the warp size is $32$, how many warps will have divergence during the fifth iteration?

    ```cpp
    __global__ void ConvergentSumReductionKernel(float* input, float* output, int N) {
        unsigned int i = threadIdx.x;

        for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
            if (threadIdx.x < stride) {
                input[i] += input[i + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *output = input[0];
        }
    }
    ```
***Solution***:

For the improved reduction kernel with 1024 elements and a warp size of 32, no warps will experience divergence during the fifth iteration. Here's the analysis:

1. **Kernel Behavior:**
    - The loop starts with `stride = blockDim.x` ($1024$ threads) and halves the stride each iteration.
    - Active threads at each iteration are `threadIdx.x < stride`.

2. **Fifth Iteration Details:**
    - Stride at fifth iteration:
    ```math 
    \text{stride} = \frac{1024}{2^4} = 64
    ```
    - Active threads: `0–63` (64 threads total).

3. **Warp Alignment:**
    - $64$ threads correspond to $2$ full warps $(64 ÷ 32 = 2)$.
    - **Both warps execute uniformly:** all threads in these warps satisfy `threadIdx.x < 64`, so no divergence occurs.

4. **Key Optimization:**
    - The improved kernel avoids partial warp activation by ensuring active threads always align with full warps until the final iteration with `stride < 32`.

---
3. Modify the above kernel to use the access pattern illustrated below:

<div align= "center">
    <img src="./images/Qn_3.png" width="500px">
    <p><b>Fig 29_01: </b><i> Question number 3 illustratory diagram</i></p>
</div>

***Solution***:
