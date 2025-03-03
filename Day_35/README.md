## Summary of Day 35: 

> *Exercises from Chapter 11 _(Page: 260 onwards)_

> ✦ _Well I just went through some questions that I felt like important ones._

### ***Exercises:***

##### Qn.1 Consider the following array: $[4 \space 6\space  7\space  1\space  2\space  8\space  5\space  2]$.Perform a parallel inclusive prefix scan on the array, using the Kogge-Stone algorithm. Report the intermediate states of the array after each step.

***Solution***

Okay, first let's solve this theoritically then the code implementation:

Here, 
**Initial Input Array**:
$$A = [4 \space 6\space  7\space  1\space  2\space  8\space  5\space  2]$$

**Step 1: *Stride 1***

The output from this step would be:

$$A' = [ 4 \space 10 \space 13 \space 8 \space 3 \space 10 \space 13 \space 7]$$

> Consult the notes if you don't get what's going on 😪.

**Step 2: *Stride 2***

$$A'' = [4 \space 10 \space 17 \space 18 \space 16 \space 18 \space 16 \space 17]$$

**Step 3: *Stride 4***

$$A''' = [4 \space 10 \space 17 \space 18 \space 20 \space 28 \space 33 \space 35]$$

_(*Stopping condition since futher strides would go out of bounds)_

> Also code implementation is here! [Click Here](./Qn1.cu) to redirect!!

##### Qn.2 Modify the Kogge-Stone parallel scan kernel in Code below to use double buffering instead of a second call to `__syncthreads()` to overcome the write after-read race condition.

```cpp
__global__ void koggeStoneScanKernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        float temp;
        if (threadIdx.x >= stride) {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();

        if (threadIdx.x >= stride) {
            XY[threadIdx.x] = temp;
        }
    }
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }
}
```

***Solution***:

So, using **double buffering** instead of second call to `__syncthreads()`, to overcome **write-after-read race condition**

The kernel would look like this:

```cpp
__global__ void koggeStoneScanKernel(float *X, float *Y, unsigned int N) {    
    __shared__ float XY[2][SECTION_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        XY[0][threadIdx.x] = X[i];
    } else {
        XY[0][threadIdx.x] = 0.0f; 
    }

    int readBuffer = 0;
    int writeBuffer = 1;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads(); 

        float temp = XY[readBuffer][threadIdx.x]; 
        if (threadIdx.x >= stride) {
            temp += XY[readBuffer][threadIdx.x - stride];
        }
        XY[writeBuffer][threadIdx.x] = temp;
        readBuffer = 1 - readBuffer;
        writeBuffer = 1 - writeBuffer;
    }
    if (i < N) {
        Y[i] = XY[readBuffer][threadIdx.x]; 
    }
}
```
***Benefits of This Approach***
1. **Improved Correctness:** The double buffering method clearly separates read and write phases, eliminating race conditions that might otherwise occur due to overlapping memory accesses.
2. **Potential Performance Gains:** Reducing the number of `__syncthreads()` calls per iteration can lower the barrier synchronization overhead, which is beneficial for performance especially on large thread blocks.
3. **Simplified Data Dependency Management:** By toggling between two buffers, the algorithm's data dependencies become clearer. Each iteration operates on a consistent snapshot of the previous results, making the algorithm easier to reason about and maintain.

> Well I GPT'd the benefits but well that's just fact ig 😅.

##### Qn.3 Analyze the same Kernel Code from Qn.2. Show that control divergence occurs only in the first warp of each block for stride values up to half of the warp size. That is, for warp size $32$, control divergence will occur to iterations for stride values $1$, $2$, $4$, $8$, and $16$.

***Solution***:

Okay, so control divergence occours in CUDA when threads within a warp _(group of 32 threads)_ take different execution paths due to conditionals. 

In the kernel above or well let's just repeat here as well:
```cpp
__global__ void koggeStoneScanKernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        float temp;
        if (threadIdx.x >= stride) {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();

        if (threadIdx.x >= stride) {
            XY[threadIdx.x] = temp;
        }
    }
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }
}
```

Well here, the control divergence occours from these conditional statements:

```cpp
if (threadIdx.x >= stride) {
    temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
}
```
and 

```cpp
if (threadIdx.x >= stride) {
    XY[threadIdx.x] = temp;
}
```
> ***Warp Basics***:
> - A warp is a group of $32$ threads that execute instructions in lockstep on CUDA hardware.
> - If threads within a warp evaluate a conditional *(if)* differently, the warp must execute both branches sequentially, causing control divergence.

So here, threads in a warp have indices `threadIdx.x = 0, ... , 31` 

For strides **up to half the warp size**, ie `stride = 1, 2, 4, 8, 16`, only some of the threads satisfy the condition `threadIdx.x >= stride`, hence they cause divergence. 

***Analysis for Stride Values***:

**Stride = 1**
- *Condition:* `if (threadIdx.x >= 1)`
- Threads $0$ do not satisfy the condition $(0 < 1)$, but threads 1–31 do.
- *Result:* Control divergence occurs because thread $0$ takes a different path than threads $1–31$.

**Stride = 2**
- *Condition:* `if (threadIdx.x >= 2)`
- Threads $0–1$ do not satisfy the condition $(0 < 2, 1 < 2)$, but threads $2–31$ do.
- *Result*: Control divergence occurs because threads $0–1$ take a different path than threads $2–31$.

**Stride = 4**
- *Condition:* `if (threadIdx.x >= 4)`
- Threads $0–3$ do not satisfy the condition $(0 < 4, ..., 3 < 4)$, but threads $4–31$ do.
- *Result:* Control divergence occurs because threads $0–3$ take a different path than threads $4–31$.

**Stride = 8**
- **Condition:** `if (threadIdx.x >= 8)`
- Threads $0–7$ do not satisfy the condition $(0 < 8, ..., 7 < 8)$, but threads $8–31$ do.
- **Result:** Control divergence occurs because threads $0–7$ take a different path than threads $8–31$.

**Stride = 16**
- *Condition:* `if (threadIdx.x >= 16)`
- Threads $0–15$ do not satisfy the condition $(0 < 16, ..., 15 < 16)$, but threads $16–31$ do.
- *Result:* Control divergence occurs because threads $0–15$ take a different path than threads $16–31$.

**Final Step: What Happens for Stride ≥ 32 *(Warp Size)* ?**
- *Condition:* For any `stride ≥ warp size` *(e.g., stride = $32$ )*, all threads in the first warp `(threadIdx.x = 0–31)` fail to satisfy the condition `(threadIdx.x < stride)`. No control divergence occurs because all threads take the same path.

<blockquote style="border-left: 6px solid #2196F3; padding: 10px;">
  ⓘ <strong>Note:</strong> The second warp <code>(threadIdx.x = 32 to 63)</code> behaves the same way but independently.
</blockquote>

##### Qn. 4. For the Brent-Kung scan kernel, assume that we have $2048$ elements. How many add operations will be performed in both the reduction tree phase and the inverse reduction tree phase?

***Solution***:

Here, \
$N$ = $2048$

First in **Brent-Kung Scan**; 

1. **Reduction Tree Phase**:

    The reduction tree performs a series of additions at each level, with the number of operations halving at each step:

    - **Level 1**: $2048/2$ = $1024$ operations
    - **Level 2**: $1024/2$ = $512$ operations
    - **Level 3**: $512/2$ = $256$ operations
    - ...and so on until Level $11$ with $1$ operation.

    The total number of operations in the reduction phase equals $(N-1)$, which for $N=2048$ is $2047$ operations.

2. **Reverse Tree Phase *(Down-Sweep)***:
- In the down-sweep phase, the algorithm propagates the partial sums back down the tree to produce the final prefix sums.
- **Key Point:** Not every node needs to be updated in this phase. For the Brent-Kung scan, the inverse phase requires fewer operations.
- The number of additions performed in the inverse phase is given by:
```math 
(N-1) - \log _2 (N)
```
- For $N=2048$, note that:
```math 
\log_2(2048) =11
```
so, inverse phase performs:
```math
2047- 11 = 2036 \space \text  {additions}
```
Hence, total additions would be:

```math
\text{Total Additions} = 2047 + 2036 = 4083 \space \text  {additions}
```

---
<div align="center">
    <b>
        End of Day_35🫡
    </b>
</div>