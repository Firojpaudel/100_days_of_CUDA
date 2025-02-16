## Summary of Day 20

> **Ending Chapter 7*

#### Tiled Convolution Using Caches for Halo Cells

In traditional tiled convolution, each block loads an input tile into shared memory, including additional halo cells from neighboring tiles to ensure correct filtering at boundaries. However, a more optimized approach leverages **L2 cache** to reduce explicit shared memory usage for halo cells.

---

### What’s Different?

- **Traditional Approach:** Load the full tile plus extra halo cells into shared memory.
- **Optimized Approach:** Load only the tile into shared memory, and fetch halo cells from L2 cache or global memory when needed.

### Key Changes and Why They Matter

- **Keeping Input & Output Tiles the Same Size:** 
    - Usually, input tiles are bigger than output tiles to store extra halo cells.
    - Here, input and output tiles are identical (e.g., `TILE_SIZE × TILE_SIZE`).
    - This keeps memory access patterns simple and predictable for CUDA.

- **Loading Only the Required Pixels into Shared Memory:**
    - Each thread loads just one pixel from the tile into shared memory.
    - Halo cells are not stored in shared memory anymore.

- **Accessing Halo Cells Directly from L2 Cache/Global Memory:**
    - If a thread needs a halo cell, it checks where the value is:
        - **If it’s inside the tile**: Use shared memory *(fast)*.
        - **If it’s in the halo region**: Fetch from ***L2 cache or global memory*** *(cached)*.
        - **If it’s outside the image**: Assume zero *(avoids invalid access)*.

### Why Is This Better?

- **Less Shared Memory Usage:** More memory available for other computations.
- **Lower Register Pressure:** CUDA kernels run more efficiently.
- **Better Thread Scheduling:** Tiles are power of 2 in size, which improves memory coalescing.

### How Does It Work in the Code?

1. Each thread block loads only the required pixels (not extra halo cells).
2. Convolution is applied using shared memory (for tile) and L2 cache or global memory (for halo).
3. The output is stored in global memory, ready for the next step.

> [Click Here](./tiled_2D_with_cache.cu) to view the modified code. Modified on yesterday.

---

***Exercises***

1. Calculate the $\text{y}[0]$ value in figure below:

    <div align="center">
        <img src= "./images/Exercise_1.png" width="500px">
        <p><b>Fig 20_01: </b><i>1D Convolution Boundary Condition Diagram</i></p>
    </div>

    _Solution:_ 

    For calulating the value of y[0], `x` should be designed as folowing:

    <div align="center">

    | 0 | 0 | 8 | 2 | 5 | 4 | 1 | 7 | 3 |
    |---|--|--|--|--|--|--|--|--|

    </div>

    Then using  the function provided; 
```math
\text{y}[0] =  0 \times 1 + 0 \times 3 + 8 \times 5 + 2 \times 3 + 5 \times 1 = 40 + 6 + 5 = 51
```

2. Consider performing a 1D convolution on array $\text{N} = \{4,1,3,2,3\}$ with filter $\text{F}= \{2,1,4\}$. What is the resulting output array?


    | Input Array (N) | 4 | 1 | 3 | 2 | 3 |
    |-----------------|---|---|---|---|---|
    | Filter (F)      | 2 | 1 | 4 |   |   |

    Here, the valid convolution output size: $\text{len}(N)- \text{len}(F) + 1 = 5-3+1 = 3$

    So, computing the values of $\text{y[0], y[1]}$ and $\text{y[2]}$:

    ```math
    \text{y[0]} = 4 \times 2 + 1 \times 1 + 3 \times 4 = 21
    \\
    \text{y[1]} = 1 \times 2 + 3 \times 1 + 2 \times 4 = 13
    \\
    \text{y[2]} = 3 \times 2 + 2 \times 1 + 3 \times 4 = 20
    ```

    Hence, the output array would look like:

    | Output Array (y) | 21 | 13 | 20 | 
    |------------------|----|----|----|

3. What do you think the following 1D convolution filters are doing?

    1. $[0 \space 1 \space 0]$

        _Answer:_ This is an **identity filter**. The filter copies the value of the center element from the input.

    2. $[0 \space 0 \space 1]$

        _Answer:_ This filter picks the value two steps behind the input and shifts it forward. For example, if the input is `[a, b, c, d, e]`, the output will be `[c, d, e, 0, 0]`.

    3. $[1 \space 0 \space 0]$

        _Answer:_ This filter picks the value two steps ahead of the input and shifts it backward. For example, if the input is `[a, b, c, d, e]`, the output will be `[0, 0, a, b, c]`.

    4. $[\frac{21}{2} \space 0 \space \frac{1}{2}]$

        _Answer:_ This filter applies a weighted sum where the first element is multiplied by $\frac{21}{2}$ and the third element by $\frac{1}{2}$. It can be used to emphasize the first element while slightly considering the third element.

    5. $[\frac{1}{3} \space \frac{1}{3} \space \frac{1}{3}]$

        _Answer:_ This is an **averaging filter**. It computes the average of the current element and its two neighbors. It is often used for smoothing or blurring the input.





