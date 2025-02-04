## Summary of Day 08:

> *The final day for Chapter 4

#### Warp Scheduling and Latency Tolerance

- **Warp Scheduling**:When threads are assigned to Streaming Multiprocessors (SMs), there are usually more threads than cores. Earlier GPUs could execute one warp’s instruction at a time, while modern GPUs handle multiple warps simultaneously.

- **Latency Tolerance _(Hiding)_**: To tolerate delays (like memory access latency), GPUs switch from a warp that's waiting to another that's ready to execute. This prevents idle time, maximizing efficiency.

    - _Analogy_ : Like a post office clerk serving the next customer while one steps aside to fill out forms—keeping productivity high.

- **Zero-Overhead Scheduling**: Unlike CPUs, GPUs store all warp states in hardware registers, enabling fast context-switching without performance penalties.

---
#### Resource Partitioning & Occupancy

- **Occupancy**: The ratio of active warps to the SM’s maximum capacity. High occupancy improves latency tolerance.
- **Dynamic Partitioning**: SM resources (registers, shared memory, thread/block slots) are dynamically allocated. This flexibility allows varying block/thread sizes but can cause underutilization in some cases:

    - _Example 1_ : If $\text{block size} = 32 \space \text{threads}$, but SM supports only $32 \space \text{blocks}$, occupancy drops to $50\%$.
    - _Example 2_ : A block size of $768 \space \text{threads}$ fills $1536 \space \text{threads}$ on an SM that supports $2048$—_leaving $512$ threads unutilized ($75\%$ occupancy)_.
---

> ***Exercises Selected Only:***

1.  Consider the following CUDA kernel and the corresponding host function that calls it:
```cpp
__global__ void foo_kernel (int* a, int* b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;   
    if (threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;
    }
    if (i % 2 == 0) { 
        a[i] = b[i] * 2;
    }
    for (unsigned int j = 0; j < 5 - (i % 3); ++j) {
        b[i] += j;
    }
}
void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel <<< (N + 128 - 1)/128, 128 >>> (a_d, b_d);
}
```
- What is the number of warps per block?
- What is the number of warps in the grid?
- For the statement on line 04:
    - How many warps in the grid are active?
    - How many warps in the grid are divergent?
    - hat is the SIMD efficiency $(\text{in}\space \%)$ of warp 0 of block 0?
    - What is the SIMD efficiency $(\text{in}\space \%)$ of warp 1 of block 0?
    - What is the SIMD efficiency $(\text{in}\space \%)$ of warp 3 of block 0?
- For the statement on line 07:
    - How many warps in the grid are active?
    - How many warps in the grid are divergent?
    - What is the SIMD efficiency $(\text{in}\space \%)$ of warp 0 of block 0?
- For the loop on line 09:
    - How many iterations have no divergence?
    - How many iterations have divergence?

_**Answer** :_ \
Here, we have: 
- Total elements $\text{(N)}$ = $1024$
- Threads per block = $128$
- Total Blocks = $\lceil \frac{1024}{128} \rceil = 8$

    - No. of warps ***per block** = $\frac{\text{Per block threads}}{\text{Threads in a warp}} = \frac{128}{32} = 4$

    - No. of warps in the grid = $\frac{\text{total threads}}{\text{threads in a warp}} = \frac{8 \times 128}{32} = \frac{1024}{32} = 32$

    - Analysis of Line $04$:
        - Per warp analysis: 
            - Each warp consists= 32 threads; analysing `threadIdx.x < 40 || threadIdx.x >= 104`.\
            <br>
            
            | Warp Index | Threads_range | Condition Evaluation | Divergence? |
            |-----------|-------------|-------------|----------| 
            | Warp 0 | 0-31 | All satisfy `threadIdx.x < 40` | No |
            | Warp 1 | 32-63 | 32-39 satisfy, 40-63 dont | Yes |
            | Warp 2 | 64-95 | None satisfy | No |
            | Warp 3 | 96-127 | 96-103 dont; 104-127 satisfy `threadIdx.x >=104` | Yes |

            Hence, 2 warps are divergent
        - **SIMD effeciency of Warp 0 of Block 0:**<br>
        Since all threads execute and there is no divergence, $\text{SIMD effeciency} = \frac{32}{32} \times 100 \% = 100\%$

        - **SIMD effeciency of Warp 1 Block 0:**<br>
            - Only the threads (32-39) execute the branch. ie., $39-32 = 8 \space \text{threads are active}$
```math
    \text{i.e., SIMD Effeciency} = \frac{8}{32} \times 100\% = 25\%
```
Simlar for `Warp 3 Block 0`. _And other questions as well._ 

2. For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

***Answer:*** Code Implementation: [Click Here](./Exercise_02.cu) to redirect!

> **_Output:_**
>```shell
>Number of blocks: 4
>Number of threads per block: 512
>Total number of threads: 2048
>```

3. For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?

***Answer:*** In the code above, there's a boundary check (`if (index < n)`) in the `vector_add` kernel.

| Property | Value |
|----------|-------| 
| Total threads in the Grid | 2048 |
| Valid Threads | 2000 _(vector length)_ |
| Extra Threads _(Inactive due to boundary)_| 48 |
| Last Active Warp | Warp 62 in the grid|
| Divergent Warps | 1 _(Warp 62)_ |

4. Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: $2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, \text{and} \space 2.9$; they spend the rest of their time waiting for the barrier. What percentage of the threads’ total execution time is spent waiting for the barrier?

***Answer:*** Code Implementation: [Click Here](./Exercise_04.cu) to redirect! 
> ***Output:***
> ```shell
> Percentage of time spent waiting: 17.08%
> ```

---
<div align="center">
    <b>
        End of Day_08🫡
    </b>
</div>





