#include <iostream>
#include <cuda_runtime.h>
#include <chrono>  // For measuring operation time

using namespace std;

#define N 32  // Defining the size of the 3D grid
#define IN_TILE_DIM 10  // Input tile dimension (including halo)
#define OUT_TILE_DIM 8  // Output tile dimension
#define HALO 1          // Halo size (1 for a 7-point stencil)

// Stencil coefficients
#define c0 0.5f
#define c1 0.1f
#define c2 0.1f
#define c3 0.1f
#define c4 0.1f
#define c5 0.05f
#define c6 0.05f

// Desigining Welcome Banner 
void display_welcome_banner() {
    cout << "*******************************************************\n";
    cout << "*         OPTIMIZED STENCIL KERNEL PROGRAM!           *\n";
    cout << "*       USING SHARED MEMORY TILING FOR SPEED          *\n";
    cout << "*******************************************************\n";
    cout << endl;
}

// CUDA kernel for tiled stencil computation
__global__ void tiled_stencil_kernel(float* in, float* out, unsigned int size) {
    // Allocating shared memory for the input tile (including halo regions)
    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    // Calculating global indices for the current thread
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - HALO;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - HALO;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - HALO;

    // Loading data into shared memory (handling boundary conditions)
    if (i >= 0 && i < size && j >= 0 && j < size && k >= 0 && k < size) {
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * size * size + j * size + k];
    } else {
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f; // Assigning zero to out-of-bound threads
    }
    __syncthreads();

    // Computing output only for threads within the valid output tile region
    if (threadIdx.z >= HALO && threadIdx.z < IN_TILE_DIM - HALO &&
        threadIdx.y >= HALO && threadIdx.y < IN_TILE_DIM - HALO &&
        threadIdx.x >= HALO && threadIdx.x < IN_TILE_DIM - HALO) {

        int global_i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - HALO;
        int global_j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - HALO;
        int global_k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - HALO;

        if (global_i >= 1 && global_i < N - 1 &&
            global_j >= 1 && global_j < N - 1 &&
            global_k >= 1 && global_k < N - 1) {

            out[global_i * N * N + global_j * N + global_k] =
                  c0 * tile[threadIdx.z][threadIdx.y][threadIdx.x]
                + c1 * tile[threadIdx.z][threadIdx.y][threadIdx.x - 1]
                + c2 * tile[threadIdx.z][threadIdx.y][threadIdx.x + 1]
                + c3 * tile[threadIdx.z][threadIdx.y - 1][threadIdx.x]
                + c4 * tile[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                + c5 * tile[threadIdx.z - 1][threadIdx.y][threadIdx.x]
                + c6 * tile[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}

int main() {
    //Displaying the welcome banner
    display_welcome_banner();

    // Allocating memory for the 3D array on host (CPU)
    size_t total_size = N * N * N; // Total number of elements
    float* h_in = (float*)malloc(total_size * sizeof(float));
    float* h_out = (float*)malloc(total_size * sizeof(float));

    // Initializing the input array with some values
    for (int i = 0; i < total_size; i++) {
        h_in[i] = static_cast<float>(i % 100) / 100.0f; // Assigning values between [0 and 1]
    }

    // Allocating memory on the GPU
    float* d_in, * d_out;
    cudaMalloc((void**)&d_in, total_size * sizeof(float));
    cudaMalloc((void**)&d_out, total_size * sizeof(float));
    
    // Copying input data from host to device
    cudaMemcpy(d_in, h_in, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initializing the output array to zero on the device
    cudaMemset(d_out, 0, total_size * sizeof(float));

    // Defining thread block and grid dimensions
    dim3 blockSize(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM); // Threads per block (including halo)
    dim3 gridSize((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                  (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                  (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    // Measuring operation time using chrono library
    auto start = chrono::high_resolution_clock::now();

    // Launching the tiled stencil kernel on the GPU
    tiled_stencil_kernel<<<gridSize, blockSize>>>(d_in, d_out, N);

    // Waiting for CUDA to finish execution
    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    
    // Calculating elapsed time in milliseconds
    chrono::duration<double, milli> elapsed_time = end - start;
    
    cout << "Stencil computation completed in " << elapsed_time.count() << " ms.\n";

    // Copying results back from device to host memory
    cudaMemcpy(h_out, d_out, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Checking a few values from the output array to ensure correctness
    cout << "Checking some values from the output array:\n";
    
    for (int i = N*N*(N/2)+N/2*N+N/2; i < N*N*(N/2)+N/2*N+N/2+10; i++) {
        cout << "h_out[" << i << "] = " << h_out[i] << endl;
        /* Printing values from a central region of the grid to ensure meaningful results */
    }

    // Freeing GPU memory
    cudaFree(d_in);
    cudaFree(d_out);

    // Freeing host memory
    free(h_in);
    free(h_out);

    return 0;
}
