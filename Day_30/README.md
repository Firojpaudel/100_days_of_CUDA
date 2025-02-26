## Summary of Day 30:

> *Starting Chapter 11

### Parallel Prefix Sum (Scan)

> *Prefix sum (scan) is a key operation in parallel computing, often used in algorithms like parallel sorting, stream compaction, and parallel dynamic programming.*

#### Types of Scans:
1. **Inclusive Scan:** Includes the current element in the sum.

```math
\text{out}[i] = \sum_{j=0}^{i} \text{in}[j]
```
2. **Exclusive Scan:** Does not include the current element in the sum.

```math 
\text{out}[i] = \sum_{j=0}^{i-1} \text{in}[j]
```
---
### Koggle-Stone Parallel Prefix Sum Algorithm
The **Kogge-Stone algorithm** is a parallel prefix sum algorithm optimized for fast execution with minimal dependencies. It works by:

- Performing $log(n)$ steps in parallel.
- Using more memory compared to sequential approaches.
- Reducing latency by allowing simultaneous calculations.

> ***Algorithm Steps***:
>
> 1. Start with the original array.
> 2. At each step, update the array using previous values at increasing power-of-two distances.
> 3. After $log(n)$ steps, the scan is complete.

---
> _[Click Here](./koggle_stone.cu) to look at the koggle stone algorithm implementation._

### The Detailed Explanation of Koggle-Stone Algo: 

