## Summary of Day 04:

Continuation of yesterday: 

#### Image Blurring Algo:

The image blur kernel code is given in the PMPP book @ page 83. I have made some adjustments to make the kernel code take in RGB image and blur it. 

Here's my kernel code:

```cpp
__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixValR = 0, pixValG = 0, pixValB = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    int offset = (curRow * w + curCol) * CHANNELS;
                    pixValR += in[offset];
                    pixValG += in[offset + 1];
                    pixValB += in[offset + 2];
                    ++pixels;
                }
            }
        }

        int offset = (row * w + col) * CHANNELS;
        out[offset] = pixValR / pixels;
        out[offset + 1] = pixValG / pixels;
        out[offset + 2] = pixValB / pixels;
    }
}
```
I mean well its not much of a difference but yeah slightly different. 

<details>
    <summary><b>Explanation to the Kernel Code:</b> <i>(Click to expand)</i></summary>
    <ul>
        <li>So, first understanding the function parameters:</li><br>
        <table>
            <tr><th>Parameters</th><th>Description</th></tr>
            <tr><td><code>unsigned char *in</code></td><td>Input image data stored in GPU memory (device memory)— 1D Array*.</td></tr>
            <tr><td><code>unsigned char *out</code></td><td>Output image data stored in GPU memory.— 1D Array*</td></tr>
            <tr><td><code>int w</code></td><td>Width of the image (in pixels).</td></tr>
            <tr><td><code>int h</code></td><td>Height of the image (in pixels).</td></tr>
        </table>
        <li>We are performing box blur where each pixel is replaced with the average color of neighboring pixels within a specified <b>blur radius.</b></li>
        <li>Then there comes <b>thread indexing</b> and <b>position calculation.</b> where <code>col</code> and <code>row</code> find the *x_position and *y_position respectively.
        <li>Next, checking if the thread is within the Image</li>
        <li>Then assigning the default values to the each Red, Green and Blue pixels to <code>0</code>.</li>
        <li><code>pixels</code> keep track of how many pixels contribute to the calculation.
        <li>
            <pre><code>for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
    for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {</code></pre>
        These two nested loops iterate over <code>(2 × BLUR_SIZE + 1) × (2 × BLUR_SIZE + 1)</code> neighborhood.
        </li>
        <li>Well, the BLUR_SIZE is set to <code>8</code>. So, the kernel checks a <code>17 × 17</code> grid around the pixel.</li>
        <li><pre><code>int curRow = row + blurRow;
int curCol = col + blurCol;
if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {</code></pre>
        This code segment is there for <b>handling the edge cases</b> where <code>curRow</code> and <code>curCol</code> represent the neighboring pixel coordinates. This boundary check ensures we do not access the pixels outside the image.
        </li>
        <li>And, then we accumulate RGB values where <code>offest = (row * w + col) * CHANNELS</code> finds the pixel location in the 1D array</li>
        <li>The values are accumulated for averaging</li>

</ul> 
</details>

To view the code implementation, [Click Here](./image_blur.cu)

**Output Comparison:**
|Before                   |  After                  |
|-------------------------|-------------------------|
|![Input image](./pika.jpg) |  ![Output Image](./blurred_output.png)|

---
#### Matrix Multiplication:

[Click Here](./matmul.cu) to view the code implemenation. 

**Output Obtained:**
```bash
Enter the width of the matrix: 4 
Enter elements of matrix M: 2 3 4 5 6 1 8 1 4 5 6 7 12 11 3 0
Enter elements of matrix N: 1 2 3 45 5 6 7 8 9 0 11 34 67 10 11 2
Result matrix P: 
388 72 126 260 
150 28 124 552 
552 108 190 438 
94 90 146 730 
```

---
#### Chapter 03_ Exercises _(Done Some of Them)_:
 1. In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.
    - Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design. — [Click Here](./Exercise_01_soln_a.cu) to access the solution.
    -  Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design. — [Click Here](./Exercise_01_soln_b.cu) to access the solution.

2.  A matrix-vector multiplication takes an input matrix $\text{B}$ and a vector $\text{C}$ and produces one output vector $\text{A}$. Each element of the output vector $\text{A}$ is the dot product of one row of the input matrix $\text{B}$ and $\text{C}$, that is, $\text{A[i]} = \sum^j \text{B[i][j]} + \text{C[j]}$. For simplicity we will handle only square matrices whose elements are single precision floating-point numbers. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. Use one thread to calculate an output vector element. — [Click Here](./Exercise_02_soln.cu) to access the solution.

---
<div align="center">
    <b>
        End of Day_04🫡
    </b>
</div>