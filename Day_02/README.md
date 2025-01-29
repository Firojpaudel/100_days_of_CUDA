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

---

> **Goin' through...**
<div align= "center">
<img src= "https://shorturl.at/iAVMb" width = "300px" />
</div>

---


