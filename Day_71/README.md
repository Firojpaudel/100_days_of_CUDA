## Summary of Day 71:

> *Im still competing today as well. 

1. Tanh implementation:

> [Click Here](./tanh1.cu) to see the implementation using manual tanh 🙂‍↕️

> [!Note]
> - Average performance: $28.74 \text{ GFLOPs}$
> - Average Runtime: $1.20 \text{ ms}$
> Device: **Tesla T4**
>
> - Average performance: $194.73 \text{ GFLOPs}$
> - Average Runtime: $0.25 \text{ ms}$
> Device: **NVIDIA H100**

2. Softmax:

> [Click Here](softmax_test.cu) to redirect towards code. 

> [!Note]
> - Average performance: $164.78 \text{ GFLOPs}$
> - Average Runtime: $0.93 \text{ ms}$
> Device: **NVIDIA H100**


3. Vect Addition:

***Approach 1:*** 

Trying loop unrolling *(4 elements per thread)*

> [Click Here](./) to redirect towards the code.

> [!Note]   
> - Average performance: $202.72 \text{ GFLOPs}$
> - Average Runtime: $0.95 \text{ ms}$
> Device: **NVIDIA H100**

***Approach 2:***

Using Shared Memory

> [Click Here](./vect_add_approach_shared_mem.cu) to redirect towards the code.

> [!Note]
> - Average performance: $166.15 \text{ GFLOPs}$
> - Average Runtime: $1.16 \text{ ms}$
> Device: **NVIDIA H100**

Using both at the same time:

> [Click Here](./vect_add_approach_both.cu) to redirect towards the code.

> [!Note]
> - Average performance: $245.66 \text{ GFLOPs}$
> - Average Runtime: $0.74 \text{ ms}$
> Device: **NVIDIA H100**
