## Summary of Day 99: 

> 1 Day left 🔥

Today's kernels:

1. **2D Average Pooling**:

> [Click Here](./2d_avg_pool.cu) to redirect to the code.

> [!note]
> - Performance: $962 \text{ GFLOPs}$
> - Runtime: $0.15 \text{ ms}$
> - GPU: **H100**

2. Optimized the previous MatMul Kernel. 

>[!tip]
> This time I used __hfma2 and half2

> [Click Here](./optimized_matmul.cu) to redirect to the code.

> [!note]
> - Performance: $38.4 \text{ TFLOPs}$ 🔥
> - Runtime: $13.31 \text{ ms}$
> - GPU: **H100**