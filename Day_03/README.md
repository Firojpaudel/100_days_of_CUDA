## Summary of Day 03:

### Reading Chapter 3: Multidimensional Grids & Data

#### MultiDimensional Grid Organization 
- A grid consists of one or more blocks and each block consits of one or more threads.
- So, in general, a grid is a 3D array of blocks and each block is a 3D array of threads.

Example program to show this: [Click here](./grids.cu) to redirect.

#### Mapping threads to a multidimensional data

Assuming, we are dealing with a picture **P** with the dimension of $62 \times 76$ and let's say the block size is $16 \times 16$,

If we were to construct the grid for this picture, assuming on x dimension, we put let's say $5$ blocks;

```math
\text{Total number of blocks}= \frac{62 \times 76}{16 \times 16} = 18.40 \sim 20\space \text{blocks}
```
Which is $5\times 4$. ie., there needs to be $5— 16 \times 16 \space\text{blocks}$ on x axis and $4 — 16\times16 \space\text{blocks}$ on y axis.
























---
> **Goin' through...**
<div align= "center">
<img src= "https://shorturl.at/iAVMb" width = "300px" />
</div>


