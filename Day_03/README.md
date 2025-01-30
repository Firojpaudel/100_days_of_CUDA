## Summary of Day 03:

### Reading Chapter 3: Multidimensional Grids & Data

#### MultiDimensional Grid Organization 
- A grid consists of one or more blocks and each block consits of one or more threads.
- So, in general, a grid is a 3D array of blocks and each block is a 3D array of threads.

Example program to show this: [Click here](./grids.cu) to redirect.

#### Mapping threads to a multidimensional data

Assuming, we are dealing with a picture **P** with the dimension of $62 \times 76$ and let's say the block size is $16 \times 16$,

Then, Calculating the grid sizes in along the dimensions:

```math
\text{Grid size in x }= \lceil \frac{\text{Image width}}{\text{Block width}} \rceil = \lceil\frac{76}{16} \rceil = \lceil4.75\rceil = 5
```
```math
\text{Grid size in y }= \lceil \frac{\text{Image height}}{\text{Block height}}\rceil = \lceil\frac{62}{16}\rceil = \lceil3.87\rceil=4
```
Hence, **total number of blocks in the grid** is: $5 \times 4 = 20$

> _**Note**_: <br>
**Clarification on CUDA’s Ordering**<br>
CUDA follows a (x, y, z) ordering for thread/block indexing, but image dimensions are usually expressed as (height × width).<br><br>- In image processing, it's common to write height × width (e.g., 62 × 76).<br>- In CUDA indexing, we think in terms of (x, y), where x is width and y is height.

#### Handling 2D Arrays in CUDA:

- CUDA C does not support direct 2D indexing for dynamically allocated arrays because the number of columns is unknown at compile time.

- Programmers must "flatten" the 2D array into a 1D array for proper memory access.

- Statically allocated arrays allow 2D indexing, but compilers internally convert them into 1D representations.

Example code for converting a RGB image to grayscale image: [Click Here](./image_color_conv.cu) to redirect.

|Before                   |  After                  |
|-------------------------|-------------------------|
|![Input image](./pika.jpg) |  ![Output Image](./output_pika.jpg)|


---
> **Goin' through Image Blurring Algo rn...**
<div align= "center">
<img src= "https://shorturl.at/iAVMb" width = "300px" />
</div>


