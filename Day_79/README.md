## Summary of Day 79:

> Quite a busy day today. 

Just wrote a kernel for **Swish activation function**:

1. Simple kernel with 4 elements per thread

> [Click Here](./swish1.cu) to redirect to the code. 

> [!note]
> - Performance: $438.30 \text{ GFLOPs}$
> - Runtime: $0.15 \text{ ms}$
> - GPU: **NVIDIA H100**

2. A bit complex one with 8 elements per thread

> [Click Here](./swish2.cu) to redirect to the code.

> [!note]
> - Performance: $408.00 \text{ GFLOPs}$
> - Runtime: $0.18 \text{ ms}$
> - GPU: **NVIDIA H100**

3. Fall back to 4 elements per thread with device `__expf()` function and 128 threads per block.. 

> [Click Here](./swish3.cu) to redirect to the code.

> [!note]
> - Performance: $452.00 \text{ GFLOPs}$
> - Runtime: $0.16 \text{ ms}$
> - GPU: **NVIDIA H100**

