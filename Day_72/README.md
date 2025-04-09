## Summary of Day 72:

> *Still grinding the leaderboards...

***Problems***:

1. **Matrix Scalar Multiplication**:

    $$C[i][j] = A[i][j] \cdot s$$

    Where, $s$ is scalar value. 

    > Taking input as a square matrix.

- **Fist Approach:** Normal with Blocksize of 32: [Click Here](./matrix_scalar_1.cu) to access the code.

> [!note]
> - Performance: $2390.3 \text{ GFLOPs}$
> - Runtime: $0.37 \text{ ms}$
> - Device: **NVIDIA H100**

- **Second Approach:** Adding shared memory in the same code: [Click Here](./matrix_scalar2.cu) to access the code.

> [!note]
> - Performance: $217.42 \text{ GFLOPs}$
> - Runtime: $0.51 \text{ ms}$
> - Device: **NVIDIA H100**
>   - Yup shared memory performed worse 😵

2. **Matrix Vector Multiplication:**

$$C[i] = \sum_{k= 0}^{K-1} A[i][k] \cdot B[k]$$

- **First Approach:** Simple: [Click Here](./mat_vect_1.cu) to view the code.

> [!note]
> - Performance: $24.87 \text{ GFLOPs}$
> - Runtime: $2.46 \text{ ms}$
> - Device: **NVIDIA H100**

- **Second Approach:** Using atomic add: [Click Here](./mat_vect_2.cu) to view the code.

> [!note]
> - Performance: $254.17 \text{ GFLOPs}$
> - Runtime: $0.27 \text{ ms}$
> - Device: **NVIDIA H100**
>   - Wayyy Better

- **Third Approach**: Shared Memory with partial sum: [Click Here](./mat_vect_3.cu) to view the code.

> [!note]
> - Performance: $761.22 \text{ GFLOPs}$
> - Runtime: $0.17 \text{ ms}$
> - Device: **NVIDIA H100**
>   - 🥳 Yayyy!! 

- **Fourth Approach:** With warp-level reduction [Click Here](./mat_vect_4.cu) to view the code.

> [!note]
> - Performance: $798.03 \text{ GFLOPs}$
> - Runtime: $0.17 \text{ ms}$
> - Device: **NVIDIA H100**
>   - 💪 Yass!! 

- **Fifth Approach:** With loop unrolling and Warp-=level reduction: [Click Here](./mat_vect_4.cu) to view the code.

> [!note]
> - Performance: $1.14 \text{ TFLOPs}$
> - Runtime: $0.13 \text{ ms}$
> - Device: **NVIDIA H100**
>   - 🤯 🤯 🤯  