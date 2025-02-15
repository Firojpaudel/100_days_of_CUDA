/*
Program to implement tiled 2D Convolution in CUDA with Halo Cells
Fixed shared memory loading pattern to prevent race conditions
*/

#include<iostream>
#include<cuda_runtime.h>

using namespace std;

#define FILTER_RADIUS 1    // 3x3 Kernel
#define TILE_SIZE 16       // Tile size for shared memory
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)  // Kernel width (3 in this case)

// Constant Memory Declaration for the filter
__constant__ float d_filter[FILTER_WIDTH * FILTER_WIDTH];

// Kernel function for tiled 2D convolution with corrected halo loading
__global__ void tiledConv(float *d_input, float *d_output, int width, int height) {
    // Shared memory dimensions including halo elements
    const int SHARED_DIM = TILE_SIZE + 2*FILTER_RADIUS;
    __shared__ float sharedMem[SHARED_DIM][SHARED_DIM];

    // Calculate global input coordinates with halo offset
    int globalY = blockIdx.y * TILE_SIZE + threadIdx.y - FILTER_RADIUS;
    int globalX = blockIdx.x * TILE_SIZE + threadIdx.x - FILTER_RADIUS;

    // Load data into shared memory with boundary checking
    if(globalY >= 0 && globalY < height && globalX >= 0 && globalX < width) {
        sharedMem[threadIdx.y][threadIdx.x] = d_input[globalY * width + globalX];
    } else {
        sharedMem[threadIdx.y][threadIdx.x] = 0.0f; // Zero-padding
    }
    __syncthreads();

    // Only central threads process convolution results
    if(threadIdx.y >= FILTER_RADIUS && threadIdx.y < SHARED_DIM - FILTER_RADIUS &&
       threadIdx.x >= FILTER_RADIUS && threadIdx.x < SHARED_DIM - FILTER_RADIUS) {
        
        float sum = 0.0f;
        #pragma unroll
        for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
            #pragma unroll
            for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
                // Apply filter with constant memory access
                sum += sharedMem[threadIdx.y + dy][threadIdx.x + dx] 
                     * d_filter[(dy + FILTER_RADIUS)*FILTER_WIDTH + (dx + FILTER_RADIUS)];
            }
        }
        
        // Calculate output coordinates
        int outY = blockIdx.y * TILE_SIZE + (threadIdx.y - FILTER_RADIUS);
        int outX = blockIdx.x * TILE_SIZE + (threadIdx.x - FILTER_RADIUS);
        
        // Boundary check for final image dimension
        if(outY < height && outX < width) {
            d_output[outY * width + outX] = sum;
        }
    }
}

int main() {
    // Define image size (must be multiple of TILE_SIZE for full coverage)
    const int width = 32, height = 32;
    const int size = width * height * sizeof(float);

    // Host memory allocation with initialization
    float *h_input = new float[width * height];
    float *h_output = new float[width * height];
    
    // Gradient pattern initialization (i + j values)
    #pragma omp parallel for
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            h_input[i*width + j] = static_cast<float>(i + j);
        }
    }

    // Device memory management
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Laplacian edge detection filter
    const float h_filter[FILTER_WIDTH * FILTER_WIDTH] = { 
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    cudaMemcpyToSymbol(d_filter, h_filter, FILTER_WIDTH*FILTER_WIDTH*sizeof(float));

    // Block and grid dimensions adjusted for halo elements
    dim3 blockDims(TILE_SIZE + 2*FILTER_RADIUS, TILE_SIZE + 2*FILTER_RADIUS);
    dim3 gridDims((width + TILE_SIZE - 1)/TILE_SIZE, 
                 (height + TILE_SIZE - 1)/TILE_SIZE);

    // Kernel launch with timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    tiledConv<<<gridDims, blockDims>>>(d_input, d_output, width, height);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Kernel execution time: " << ms << "ms\n";

    // Result verification
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    /* Expected output pattern verification:
       - All interior pixels (1 ≤ y ≤ 30, 1 ≤ x ≤ 30) should be 0
       - Border pixels show valid convolution results */
    bool valid = true;
    for(int y = 1; y < height-1; y++) {
        for(int x = 1; x < width-1; x++) {
            if(abs(h_output[y*width + x]) > 1e-6) {
                valid = false;
                break;
            }
        }
    }
    cout << "Validation: " << (valid ? "Passed" : "Failed") << endl;

    //Printing the filtered *image 
    cout << "Filtered Output Image:\n";
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << h_output[i * width + j] << " ";  // Print each pixel value
        }
        cout << "\n";  // Newline after each row
    }
    cout << endl;

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}