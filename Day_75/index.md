## Summary of Day 75:

I tried to rethink on yesterday's Prefix Sum *(Inclusive Scan)* approach and well used Kogge Stone algo. This approach got me $11.24$ GFLOPS plus runtime of $0.07$ ms on H100 GPU.

> **Code:** [Click Here](https://github.com/Firojpaudel/100_days_of_CUDA/blob/main/Day_75/prefix_sum_inclusive_kogge.cu) to redirect.

Next tried Diagonal Matrix Multiplication:

$$ C[i][j] = A[i] \cdot B[i][j] $$
>[!note] 
>Input: 
>- Diagonal $A$ of size $N$
>- Matrix $B$ of size $N \times M$
>
>Output: 
>- Matrix $C$ of size $N \times M$

> **Code:** [Click Here](https://github.com/Firojpaudel/100_days_of_CUDA/blob/main/Day_75/diagonal.cu) to redirect.

> [!important]
> ***Benchmark:***
> - $GFLOPs$: $297.75$ $GFLOPs$
> - Runtime: $0.13$ ms

Finally, ELU kernel: *(Exponential Linear Unit)*

$$ C[i][j] = \begin{cases} A[i][j] & \text{if } A[i][j] > 0 \\ \alpha \cdot (e^{A[i][j]} - 1) & \text{if } A[i][j] \leq 0 \end{cases} $$

The ELU function is defined as:

```
f(x) =
\begin{cases} 
x & \text{if } x > 0 \\ 
\alpha \cdot (e^x - 1) & \text{if } x \leq 0 
\end{cases}
```

Where $\alpha$ is a parameter controlling the values to which an ELU saturates for negative inputs.

> **Code:** [Click Here](https://github.com/Firojpaudel/100_days_of_CUDA/blob/main/Day_75/elu.cu) to redirect.

> [!important]
> ***Benchmark:***
> - $GFLOPs$: $1301$ $GFLOPs$ $\sim 1.3\space TFLOPs$
> - Runtime: $0.19$ ms
