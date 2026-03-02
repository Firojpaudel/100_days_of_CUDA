## Summary of Day 90:

First; wrote a kernel for L2 Normalization 

> [Click Here](https://github.com/Firojpaudel/100_days_of_CUDA/blob/main/Day_90/l2_norm.cu) to redirect to the code.

> [!note]
> - Performance: $221.41 \text{ GFLOPs}$
> - Runtime: $0.17 \text{ ms}$
> - GPU: **NVIDIA H100**

Next, tried to optimize KL divergence code a bit more: Using early exit and dynamic thread block size.

> [Click Here](https://github.com/Firojpaudel/100_days_of_CUDA/blob/main/Day_90/kl_optimized.cu) to redirect to the code.

> [!note]
> - Performance: $1.4 \text{ TFLOPs}$
> - Runtime: $0.16 \text{ ms}$
> - GPU: **NVIDIA H100**
