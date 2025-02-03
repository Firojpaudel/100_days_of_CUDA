## Summary of Day 07:

> *Continuation of Control Divergence

##### Impact of Divergence on Performance:

- **More passes = More Time**: Divergence increases execution time because the GPU must process paths sequentially _(on Pascal and earlier architectures)_.

##### Examples of divergence:

1. If-Else Divergence

Example Code : [Click here](./if-else_diverge.cu) to redirect.

**_Output of the code:_**
```shell
Thread 0: Value = 0
Thread 1: Value = 3
Thread 2: Value = 4
Thread 3: Value = 9
Thread 4: Value = 8
Thread 5: Value = 15
Thread 6: Value = 12
Thread 7: Value = 21
Thread 8: Value = 16
Thread 9: Value = 27
```

> _**How this creates the divergence:**_
> - CUDA schedules threads in warps _(10 in this code case)_
> - When the if-else condition is checked:
>   - Half of the warp _(even threads)_ take one path.
>   - The other half _(odd threads)_ take another path.

2. For-Loop Divergence

Example implementation of Code: [Click Here](./loops_warp_divergence.cu) to redirect.

_**Output of the above implementation code:**_

```shell
Thread  0: Iterations = 1, Sum = 0
Thread  1: Iterations = 2, Sum = 10
Thread  2: Iterations = 3, Sum = 20
Thread  3: Iterations = 4, Sum = 30
Thread  4: Iterations = 5, Sum = 40
Thread  5: Iterations = 1, Sum = 50
Thread  6: Iterations = 2, Sum = 60
Thread  7: Iterations = 3, Sum = 70
Thread  8: Iterations = 4, Sum = 80
Thread  9: Iterations = 5, Sum = 90
```
> **_A bit of explanation:_**
> 1. **Iteration Count**:
>- Threads calculate iterations = (tid % 5) + 1.
>   - For tid = 0, iterations = 1
>   - For tid = 1, iterations = 2 \
>       ...
>   - For tid = 4, iterations = 5, and so on _(repeating > every 5 threads)._
> 2. **Divergence**:
> - All threads start the loop together.
> - Threads with fewer iterations finish early, while others continue looping.
> - This causes divergence because the warp must wait until all threads complete their loops.
> 3. **Sum Calculation:**
> - Inside the loop, each thread adds its tid to sum in > each iteration.

##### Identifying the potential Divergence:
- **Based on Conditions**: If conditions depend on `threadIdx` or `blockIdx`, divergence is likely.
- **Boundary Conditions**: Common in data processing tasks where $\text{data size} ≠ \text{block size}$ _(e.g., disabling extra threads when data runs out)_.

##### Performance Impact Analysis:

- **Small Data Sizes**: Higher performance impact because more warps are affected.
- **Large Data Sizes**: Impact is minimal as only a few warps experience divergence.

    - _**Example**:_ For a vector size of $1000$, divergence affects $\sim3\%$ of execution time.
    - For image processing, performance loss drops below $2\%$ as image dimensions grow.

##### So, how do you handle divergence correctly?
Using **Barrier Synchronization** _(like `_syncwarp()`)_ to ensure all threads complete divergent paths before procedding when necessary. 


---
> **Goin' through...**
<div align= "center">
<img src= "https://shorturl.at/iAVMb" width = "300px" />
</div>
