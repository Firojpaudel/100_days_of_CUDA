## Summmary of Day 02:

### 1. Parameters passing in `.cu`:

Well I've written a program for addition in kernel:
```cpp
#include <iostream>
using namespace std;

__global__ void add (int a, int b, int *c){
    *c = a + b;
}

int main (void){
    int c;
    int *dev_c;
    cudaMalloc((void**)&dev_c, sizeof(int));

    add <<<1,1>>> (2,7, dev_c);

    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "2 + 7 = " << c << endl;

    cudaFree(dev_c);
    return 0;
}
```

<details>
    <summary>
    <b>Explanation of above code: </b><i> Click to view </i>
    </summary>
    <ul>
    <li>Parameters passing to a kernel is similar to that of  host.</li>
    <li>In order to return value to the host, we must allocate the memory.</li>
    <li><i>If you want to store any values to the GPU; must make a pointer first.</i></li>
    <li><code>cudaMalloc((void**)&dev_c, sizeof(int));</code>
    <ul>
    <li><code>cudaMalloc</code> allocates memory to the GPU</li>
    <li><code>(void**)&dev_c</code> casts a pointer to <code>void**</code> as required by <code>cudaMalloc</code></li>
    <li><code>sizeof(int)</code> used to allocate the required space to store one integer (4bits)</li>
    </ul>
    <li><code>add<<<1,1>>>(2,7, dev_c)</code> is similar to function call; where the extra bit is the cuda kernel launch config.</li>
    <li><code>cudaMemcpy</code> copies data between host and device where <code>&c</code> is the destination address of the host, <code>dev_c</code> is the source address on the device(GPU), <code>sizeof(int)</code> specifies the number of bytes to copy, and finally <code>cudaMemcpyDeviceToHost</code> specifies the direction of copy</li>
    </li>
    </ul>
</details>

---

### 2. Device Queries:

Getting to know all the stats related to device (GPU):

Below are the properties and their descriptions:

| **CUDA Device Properties** | _Description_ |
|---|---|
| `name[256]` | Device name string. |
| `totalGlobalMem` | Global memory size in bytes. |
| `sharedMemPerBlock` | Shared memory per block in bytes. |
| `regsPerBlock` | Number of 32-bit registers per block. |
| `warpSize` | Warp size in threads. |
| `memPitch` | Maximum pitch allowed by memory copy in bytes. |
| `maxThreadsPerBlock` | Maximum threads per block. |
| `maxThreadsDim[3]` | Maximum dimension of each block (x, y, z). |
| `maxGridSize[3]` | Maximum dimension of each grid (x, y, z). |
| `totalConstMem` | Constant memory size in bytes. |
| `major` | Major compute capability. |
| `minor` | Minor compute capability. |
| `clockRate` | Clock frequency in kHz. |
| `textureAlignment` | Alignment requirement for texture data. |
| `deviceOverlap` | Whether the device can overlap kernel execution and memory transfers. |
| `multiProcessorCount` | Number of multiprocessors on the device. |
| `kernelExecTimeoutEnabled` | Timeout for kernels (1 if enabled). |
| `integrated` | Whether the GPU is integrated. |
| `canMapHostMemory` | If the device can map host memory. |
| `computeMode` | Device compute mode. |
| `maxTexture1D` | Maximum 1D texture size. |
| `maxTexture2D[2]` | Maximum 2D texture size (width, height). |
| `maxTexture3D[3]` | Maximum 3D texture size (width, height, depth). |
| `maxTexture2DArray[3]` | Maximum dimension of layered 2D textures. |
| `concurrentKernels` | Number of concurrent kernels. |

Exmaple code: [Click here](./dev_queries.cu) to redirect to `dev_queries.cu` file.

---

> *Read till chapter 3 of this book. Now will shift to the PMPP book. Since it was recommended* 🤷.

*Will start from **Chapter 2; section 2.3**—looks like a continuation from here onwards*🙂‍↕️

Okay so bofore I went through parameters assignmet. Now, time for "Vector Addition Kernel"

Redirect to code by clicking [here](./vect_addn.cu)

It's pretty similar to the one done before. 
However explaining the reasoning behind some code snippets:

```cpp
__global__ void vectadd_kernel(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}
```
_**Parameters:**_
- `float *A` , `float *B`: are the pointers to the input vectors in GPU.
- `float *C` : is the pointer to the output vector.
- `int n`: The number of elements in each vector.

Likewise, `blockIdx.x * blockDim.x` calculates the starting index for the current block and adding `threadIdx.x` gives us the global index.

And finally, putting boundary check to prevent *out_of_bounds memory access*.

---

_Chapter 2: Exercises_

1.  If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?
    -  *Answer* : `i=blockIdx.xblockDim.x + threadIdx.x;` _option C_ 
    > _Just did earlier on vector addition as well_ 🤓.

2. Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?
    - *Answer* : `i=blockIdx.xblockDim.x2 + threadIdx.x;` _option D_

3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes `2blockDim.x` consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?
    - *Answer* : `i=(blockIdx.x×blockDim.x+threadIdx.x)×2`
    - *Reasoning* : Each thread processes 2 elements of a vector, but instead of processing them consecutively(like in Qn.2), all threads in a block first process one section before moving to the another one. <br>
    So, this means that each block processes $2 \times \text{blockDim.x}$ elements, each thread processes **2 elements**, but in different passes. Hence, leading to the answer provided.  

4.  For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements.How many threads will be in the grid?
    - *Answer* : 8192 
    - *Reasoning* : $\text{num_blocks} = \frac{\text{TotalVectLen}}{\text{ThreadBlockSize}}$. So, that means: $$\text{num_blocks} = \frac{8000}{1024}= 7.8 \sim 8 \text{blocks}$$
    Hence, total number of threads would be:
    $$\text{num_blocks} \times \text{blockDim.x} = 8 \times 1024 = 8192$$

5. If we want to allocate an array of v integer elements in the CUDA device `global` memory, what would be an appropriate expression for the second argument of the `cudaMalloc` call?
    - *Answer* : `v * sizeof(int)`

6.  If we want to allocate an array of `n` floating-point elements and have a floating-point pointer variable `A_d` to point to the allocated memory, what would be an appropriate expression for the first argument of the `cudaMalloc()` call?
    - *Answer* : `cudaMalloc((void**)&A_d, n* sizeof(float))`. So, it would be `(void**)&A_d`

7.  If we want to copy 3000 bytes of data from host array `A_h` (`A_h` is a pointer to element 0 of the source array) to device array `A_d` (`A_d` is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?
    - *Answer* : A typical `cudaMemcpy` syntax would look like:
    ```cpp 
    cudaMemcpy(destination, src, size, direction)
    ```
    Hence, the answer is: `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`

8. How would one declare a variable err that can appropriately receive the returned value of a CUDA API call?
    - *Answer* : `cudaError_t err;`

9. Consider the following CUDA kernel and the corresponding host function that calls it:

```cpp
 __global__ void foo_kernel(float a, float b, unsigned int N){
    unsigned int i=blockIdx.xblockDim.x + threadIdx.x;
    if(i , N) {
        b[i]=2.7fa[i]- 4.3f;
    }
}

void foo(float a_d, float b_d) {
    unsigned int N=200000;
    foo_kernel <<<(N + 128-1)/128, 128>>>(a_d, b_d, N);
 }
```
a. What is the number of threads per block? — *Answer*: `128 threads per block`
> Reasoning: 
```cpp
kernel_name<<<number_of_blocks, number_of_threads_per_block>>>(args);
```

b. What is the number of threads in the grid?
$$
\text{numblocks} = \frac{N + 128 -1}{128} = \frac{200000 + 128 -1}{128} = 1562 \text{blocks}
$$
So, 
$$
\text{totalthreads} = \text{numblocks} \times \text{threads per block} = 1562 \times 128 = 200000 \text{threads}
$$

c.  What is the number of blocks in the grid?
_Calculated earlier_: `1562 blocks`

d. What is the number of threads that execute the code on line 02?

Answer would be the $\text{totalthreads}$ ie., $200,000$

e. What is the number of threads that execute the code on line 04?

$$\text{i ranges from} = 0 \space \text{to} \space \text{N}-1$$
$$\text{N} = 200000$$
the code below checks for iterations in which i is less than N. ie., $0- (200000-1) \space \text{times}$
```cpp
if(i , N) {
        b[i]=2.7fa[i]- 4.3f;
    }
```
*Answer* : `200000 threads`

10. A new summer intern was frustrated with CUDA. He has been complaining
 that CUDA is very tedious. He had to declare many functions that he plans
 to execute on both the host and the device twice, once as a host function and once as a device function. What is your response? 

    - *Answer* : Using `__host__` and `__device__` qualifiers 🤷.

---
<div align="center">
    <b>
        End of Day_02 🫡
    </b>
</div>


