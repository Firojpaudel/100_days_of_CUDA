# 100_days_of_CUDA
Challenging myself to learn CUDA (Basics ---> Intermediate) in this 100 Days about the CUDA. 

My learning resources: 
1. **Books**:
    - **Cuda By Example** _An Introduction to General-Purpose GPU Programming_ — Jason Sandres, Edward Kandrot
    - **PMPP**; _*4th Edition_ — Wen-mei, David, Izzat
#### Progress: 

<div align="center">

| Days  | Learnt Topics | Link/s |
|-------|---------------|------|
| Day 01 | History, Applications, Setup, First Hello World CUDA program | [🔗 README](./Day_01/README.md), [🔗 Code](./Day_01/hello.cu) |
| Day 02 | Parameters Passing, Device Queries, Vector Addition on Kernel, PMPP Chapter_02 Exercises Solved| [🔗 README](./Day_02/README.md), [🔗Params_passing Code](./Day_02/params.cu), [🔗Dev_Query Code](./Day_02/dev_queries.cu), [🔗Vect_addn Code](./Day_02/vect_addn.cu)|
| Day 03 | Multidimensional Grids Organization, Mapping Threads to Multidimensional Data | [🔗README](./Day_03/README.md), [🔗Code for Grids Explanation](./Day_03/grids.cu), [🔗RGB to Grayscale Conversion Code](./Day_03/image_color_conv.cu) |
| Day 04 | Image Blurring, Matrix Multiplication, Solution to Exercise 1(a,b) and 2 | [🔗README](./Day_04/README.md), [🔗Image Blurring Code](./Day_04/image_blur.cu), [🔗Exercise_Qn.1(a) Code](./Day_04/Exercise_01_soln_a.cu), [🔗Exercise_Qn.1(b) Code](./Day_04/Exercise_01_soln_b.cu), [🔗Exercise 2 Code](./Day_04/Exercise_02_soln.cu)|
| Day 05 | Architecture of modern GPU _(intro)_, Architecture diagram understanding, Block Scheduling, Barrier Synchronization and using `__syncthreads()`| [🔗README](./Day_05/README.md), [🔗Barrier Synchronization Code](./Day_05/barrier_sync.cu) |
| Day 06 | Warps and SIMD Hardware, The modern GPU architecture, Control Divergence _Intro_ | [🔗README](./Day_06/README.md) | 
| Day 07 | Studying impacts of divergence on Performance, Types of divergence _(If-else & Loop-Based)_, Divergence identification, Performance Impace Analysis | [🔗README](./Day_07/README.md), [🔗if_else divergence Code](./Day_07/if-else_diverge.cu), [🔗loops_warp_divergence Code](./Day_07/loops_warp_divergence.cu) |
| Day 08 | Warp Scheduling and Latency Tolerance, Resource Partitioning & Occupancy | [🔗README](./Day_08/README.md), [🔗Exercise_02_Solution Code](./Day_08/Exercise_02.cu), [🔗Exercise_04_Solution Code](./Day_08/Exercise_04.cu) |
</div>
