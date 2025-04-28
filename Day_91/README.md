## Summary of Day 91:

Kernel's of the day:
1. Symmetric Matrix Multiplication:

***Kernel: 1***

> [Click Here](symm_mat_mul_1.cu) to redirect towards code

> [!note]
> - Performance: $4.8 \text{ TFLOPs}$
> - Runtime: $166.12 \text{ ms}$
> - GPU: **A100-80GB**

***Kernel: 2***

> [Click Here](symm_mat_mul_2.cu) to redirect towards code


> [!note]
> - Performance: $8.5 \text{ TFLOPs}$
> - Runtime: $93.81 \text{ ms}$
> - GPU: **A100-80GB**

> [!caution]
> For some reason, I could not get it working on H100 GPUs. It failed in last test case. I'll try to optimize that in the future

2. Gemm with bias and RELU:

> [Click Here](gemm_bias_relu.cu) to redirect towards code

> [!important]
> I went through HAMDI's code for this. I just could not get it right on first time. Will try to give my own touch some day. I optimized his code basically 

> [!note]
> - Performance: $10.6 \text{ TFLOPs}$
> - Runtime: $1.43 \text{ ms}$
> - GPU: **A100-80GB**