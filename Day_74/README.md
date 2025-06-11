## Summary of Day 74:

> Back to Tensara Grind!

So, today I'll try to code the kernel for cumulative sum (aka. prefix sum or scan) if a input array:

$$\text{output}[i] = \sum_{j=0}^{i} \text{input}[j]$$


Input: Vector $\text{input}$ of size $N$.
Output: Vector $\text{output}$ of size $N$ containing cumulative sums.

>[!note]
> The first element of the output is equal to the first element of the input

$1^{st}$ ***approach***: Naive CUDA Kernel

> [Click Here](./Naive_cumu.cu) to redirect to the code.

>[!caution]
> This gave me just 0.01 GFLOPs 💀 

$2^{nd}$ ***approach***: Naive with Shared Memory

> [Click Here](./shared.cu) to redirect to the code.

>[!warning]
> Just a bit of improvement *(just passes the benchmarks)* 0.03 GFLOPs 😵

$3^{rd}$ ***approach***: Multi kernel Approach

> [Click Here](./multi_kernel.cu) to redirect to the code.

>[!warning]
> A bit of improvement *(just passes the benchmarks)* ~3 GFLOPs 😵


Giving up for today on this... 

Next Question: **Average Pooling 1D**

> [Click Here](./average_pooling.cu) to redirect to the code.

Question 3: $1D$ **Max Pooling**

> [Click Here](./max_pooling.cu) to redirect to the code.


