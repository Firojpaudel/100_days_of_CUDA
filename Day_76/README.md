## Summary of Day 75:

Trying out cummulative product today.

$$\text{output}[i] = \prod_{j=0}^{i} \text{input}[j]$$ 

Assumption both vectors are of same sizes. ie., $N$

$1^{st}$ ***approach:*** Naive simplest kernel:

>[Click Here](./naive_1.cu) to redirect to the code.

The performance of this one was very bad. Like very very bad. 

> [!note]
> The best it could go was up to:
> - Performance: $0.03 \text{ GFLOPs}$
> - Runtime: $0.03 \text{ ms}$

$2^{nd}$ ***approach:*** Multikernel approach

> [Click Here](./multikernel.cu) to redirect to the code.

> [!note]
> - Perfromance: $5.5 \text{ GFLOPs}$
> - Runtime: $0.10 \text{ ms}$


--- 
> Tried other ways but idk why its failing. Will look into this tomorrow! 


