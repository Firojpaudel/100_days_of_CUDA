## Summary of Day 10:

> _*Chapter 5 continuation_

#### CUDA Memory Types

CUDA devices feature multiple types of memory, each with unique characteristics in terms of latency, bandwidth, scope, and lifetime. These memory types are designed to optimize the compute-to-memory access ratio, enabling efficient parallel computation. Here's a detailed breakdown:

1. **Global Memory:**
- **Location**: Off-Chip DRAM.
- **Access**: Read/Write by host and device.
- **Latency**: High (long access latency). 
- **Bandwidth**: Relatively low compared to on-chip memory.
- **Scope**: Accessible by all threads accross all grids and also by the host.
- **Lifetime**: Entire application execution.
- **Usage**:
    - Used for large datasets that need to be shared across all threads.
    - Slower to access but can be improved with caching in modern GPUs.

_Example:_
```cpp
__device__ float globalVar; //Global Variable declaration 

__global__ void kernelFuncn(){
    globalVar = 10.0; // Accessing the global memory
}
```
2. **Constant Memory:**
- **Location**: Off-Chip (cached)
- **Access**: Read-only by device, host can write
- **Latency**: Low (short-latency reads when cached).
- **Bandwidth**: High if accessed effeciently.
- **Scope**: Read-only for all threads across all grids.
- **Lifetime**: Application duration.
- **Performance**: Fast for read-only, cached.
- **Usage**:
    - Ideal for variables that remain constant throughout execution _(e.g., configuration parameters)_.

_Example:_
```cpp
__constant__ float constVar; // Constant variable declaration

__global__ void kernelFunction() {
    float value = constVar; // Accessing constant memory
}
```

3. **Local Memory:**
- **Location**: Stored in global memory _(not truly local)_
- **Latency**: Similar to global memory _(high latency)_
- **Scope**: Private to each thread
- **Lifetime**: Duration the thread's execution
- **Usage**:
    - For data private to a thread that cannot fit into registers _(eg. spilled registers or large arrays)_

_Example:_
```cpp
__global__ void kernelFunction() {
    int localArray[10]; // Stored in local memory
    localArray[0] = threadIdx.x; 
}
```

4. **Registers:**
- **Location**: On-chip register file.
- **Latency**: Extremely low _(shortest access latency)_.
- **Bandwidth**: Highest among all memory types.
- **Scope**: Private to each thread.
- **Lifetime**: Duration of the thread's execution.
- **Usage**:
    - For frequently accessed variables within a thread.

_Example:_
```cpp
__global__ void kernelFunction() {
    int registerVar = threadIdx.x; // Stored in registers
}
```
> _Slight Notes:_
> - Registers consume significantly less energy compared to global memory, making them highly efficient.
> <br>
> - Excessive use of registers can reduce the number of active threads per SM, affecting occupancy and performance.

5. **Shared Memory:**
- **Location**: On-chip scratchpad memory within an SM _(Streaming Multiprocessor)_.
- **Latency**: Low _(higher than registers but much lower than global memory)_.
- **Bandwidth**: High when accessed efficiently.
- **Scope**: Shared among all threads within a block.
- **Lifetime**: Duration of the block's execution.
- **Usage**:
    - For collaboration among threads within a block _(eg. intermediate results and shared data)_.

_Example:_
```cpp
__shared__ float sharedVar[256]; // Shared variable declaration

__global__ void kernelFunction() {
    sharedVar[threadIdx.x] = threadIdx.x; // Accessing shared memory
    __syncthreads(); // Synchronize threads within the block
}
```
> _**Note:**_
Use `__syncthreads()` to ensure proper synchronization among threads in a block when accessing shared memory.
---
### **TLDR;**
| Memory Type | Location | Latency | Bandwidth | Scope | Lifetime|
|-------|------|-----|----|------|-----|
| Global Memory|	Off-chip DRAM	|High	|Low	|All threads & host	|Entire application|
|Constant Memory|	Global + Cache|	Low _(cached)_	|High	|All threads	|Entire application|
|Local Memory	|Global Memory|	High	|Low	|Private to a thread	|Thread's execution|
|Registers	|On-chip	|Very low	|Very high	|Private to a thread	|Thread's execution|
|Shared Memory	|On-chip	|Low	|High	|Threads in a block	|Block's execution|

**Variable Declaration Syntax in CUDA:**
- `__device__`: Declares global variables.
- `__shared__`: Declares shared variables.
- `__constant__`: Declares constant variables.

[Click Here](./mem_types_in_action.cu) to view the code that implements all of these in one spot.

> **How this Explains CUDA Memory Types:** 
><br><br>
> _From the code:_

| **Memory Type**| **Where It Appears in Code** |**Explanation**|
|----------------|------------------------------|---------------|
| **Global Memory**     | `float *globalmem`, `cudaMalloc(&d_input, bytes)`, `cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice)` | - The input array (`d_input`) and output array (`d_output`) are allocated in global memory. <br> - The kernel reads data from global memory (`globalmem[idx]`) and writes results back to global memory (`output[idx]`).              |
| **Constant Memory**   | `__constant__ float const_data;`, `cudaMemcpyToSymbol(const_data, &scaleValue, sizeof(float));`             | - The constant array `const_data` is used to store the scaling factor (`2.5`). <br> - All threads read this value with low latency using the constant memory cache.                                                                 |
| **Shared Memory**     | `__shared__ float shared_data[256];`, `shared_data[threadIdx.x] = globalmem[idx];`                             | - Shared memory (`shared_data`) is used to temporarily store data from global memory for faster access. <br> - Threads within a block share this data to perform computations collaboratively.                                       |
| **Local Memory**      | `float localVar = 0.0f;`, `localVar = regVar + 1.0f;`                                                          | - Each thread has its own private variable (`localVar`) stored in local memory. <br> - This variable stores intermediate results that are specific to the thread and not shared with others.                                         |
| **Registers**         | `float regVar = 0.0f;`, `regVar = shared_data[threadIdx.x] * const_data;`                                   | - The variable `regVar` is stored in registers for fast access. <br> - It holds frequently used intermediate values during computation (e.g., scaled values). Registers are the fastest type of memory in CUDA.                      |

---
<div align="center">
    <b>
        End of Day_10🫡
    </b>
</div>

