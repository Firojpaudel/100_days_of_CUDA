## Summary of Day 11:

> _Okay so yesterday, I explored the different types of memory available, such as global memory (large but slow) and shared memory (small but fast). This understanding laid the foundation for learning about the tiling concept, which optimizes memory usage and improves computational efficiency._

#### Tiling Concept and Memory Tradeoff:
- CUDA programming involves a **tradeoff** _between global memory and shared memory_:
    - **Global Memory**: Large in size but has high latency and slower access.
    - **Shared Memory**: Limited in size but offers low latency and faster access.
- **Tiling** is a strategy to partition large datasets into smaller subsets, called tiles, that fit into shared memory. This reduces global memory traffic by enabling threads to collaborate and reuse data stored in shared memory.


_**Example case:**_

##### **Matrix Multiplication**
1. **Global Memory Access Without Tiling**:
    - Threads redundantly access overlapping elements of matrices $M$ and $N$ from global memory.
    - *For example*, multiple threads might repeatedly load the same row or column elements, leading to inefficiency.

2. **Tiled Matrix Multiplication Algorithm:**
    - The input matrices $M$ and $N$ are divided into smaller tiles that fit into shared memory.
    - Threads within a block collaboratively load these tiles into shared memory arrays (`Mds` for $M$ and `Nds` for $N$).
    - Each thread uses the tile data to compute partial dot products, which are accumulated over multiple phases.

3. **Execution Phases:**
    - Computation is divided into phases:
        - In each phase, threads load a tile of $M$ and $N$ into shared memory.
        - The loaded tiles are used to calculate partial results for the output matrix.
    - *For example*, with a tile size of $2×2$, threads load specific elements of $M$ and $N$ into shared memory, perform calculations, and repeat this process until the entire matrix is processed.
4. **Reduction in Global Memory Traffic:**
    - By collaborating, threads ensure that each element of $M$ and $N$ is loaded from global memory only once.
    - The reduction in global memory traffic is proportional to the tile size. For instance:
        - With $16×16$ tiles, global memory traffic can be reduced to $1/16$ of the original level.

[Click Here](./tiled_mat_mul.cu) to view the full implementation of tiled matrix multiplication.


---
<div align="center">
    <b>
        End of Day_11🫡
    </b>
</div>


