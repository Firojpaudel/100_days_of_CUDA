## Summary for Day 95:

So, today was al about the 2D Convolution Kernel:

1. Implementation 1: 

> [Click Here](./2d_conv_1.cu) to redirect to the code.

> [!note]
> - Performance: $2.93 \text{ TFLOPs}$
> - Runtime: $35.42 \text{ ms}$
> - GPU: **H100**

2. Implementation 2: Naive Kernel *yeah! this performed surprisingly well*

> [Click Here](./2d_conv_2.cu) to redirect to the code.

> [!note]
> - Performance: $3.29 \text{ TFLOPs}$
> - Runtime: $31.24 \text{ ms}$
> - GPU: **H100**

3. Implementation 3: Optimized Kernel

> [Click Here](./2d_conv_3.cu) to redirect to the code.

> [!note]
> - Performance: $3.45 \text{ TFLOPs}$
> - Runtime: $35.31 \text{ ms}$
> - GPU: **H100**