#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

using namespace std;

// **CUDA Error Checking Macro**
#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        cudaError_t error = call;                                                                   \
        if (error != cudaSuccess) {                                                                 \
            cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << endl; \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

// **Utility Function to Print Arrays**
void printArray(float* arr, int n, const char* label) {
    cout << "\n" << label << ":\n";
    cout << fixed << setprecision(2);
    for (int i = 0; i < n; i++) {
        cout << "  [" << setw(2) << i << "] = " << setw(8) << arr[i];
        if (i < n - 1) cout << "\n";
    }
    cout << endl;
}

// **Function to Print Matrix in Dense Format**
void printHybridMatrixDense(int numRows, int numCols, int maxNonzerosPerRow, int numELLRows,
                            unsigned int* h_colIdxELL, float* h_valueELL,
                            int numCOONonzeros, unsigned int* h_rowIdxCOO, unsigned int* h_colIdxCOO, float* h_valueCOO) {
    cout << "\nSparse Matrix A in dense format:\n";
    cout << fixed << setprecision(2);

    // Create a dense representation initialized to zero
    float* dense = new float[numRows * numCols]();

    // Fill ELL part
    for (int row = 0; row < numELLRows; row++) {
        for (int i = 0; i < maxNonzerosPerRow; i++) {
            unsigned int col = h_colIdxELL[row * maxNonzerosPerRow + i];
            if (col < numCols) {
                dense[row * numCols + col] = h_valueELL[row * maxNonzerosPerRow + i];
            }
        }
    }

    // Fill COO part
    for (int idx = 0; idx < numCOONonzeros; idx++) {
        unsigned int row = h_rowIdxCOO[idx];
        unsigned int col = h_colIdxCOO[idx];
        if (row < numRows && col < numCols) {
            dense[row * numCols + col] = h_valueCOO[idx];
        }
    }

    // Print the dense matrix
    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < numCols; col++) {
            cout << setw(8) << dense[row * numCols + col];
        }
        cout << endl;
    }

    delete[] dense;
}

// **Function to Print Calculation Process for Verification**
void printCalculationProcess(int numRows, int numCols, int maxNonzerosPerRow, int numELLRows,
                             unsigned int* h_colIdxELL, float* h_valueELL,
                             int numCOONonzeros, unsigned int* h_rowIdxCOO, unsigned int* h_colIdxCOO, float* h_valueCOO,
                             float* h_x, float* h_y) {
    cout << "\nCalculation process for y = A * x:\n";
    cout << fixed << setprecision(2);

    for (int row = 0; row < numRows; row++) {
        cout << "y[" << row << "] = ";
        float sum = 0.0f;
        bool first = true;

        // ELL contributions
        if (row < numELLRows) {
            for (int i = 0; i < maxNonzerosPerRow; i++) {
                unsigned int col = h_colIdxELL[row * maxNonzerosPerRow + i];
                if (col < numCols) {
                    float value = h_valueELL[row * maxNonzerosPerRow + i];
                    float term = value * h_x[col];
                    sum += term;
                    if (!first) cout << " + ";
                    cout << value << " * " << h_x[col] << " (A[" << row << "][" << col << "] * x[" << col << "])";
                    first = false;
                }
            }
        }

        // COO contributions
        for (int idx = 0; idx < numCOONonzeros; idx++) {
            if (h_rowIdxCOO[idx] == row) {
                unsigned int col = h_colIdxCOO[idx];
                float value = h_valueCOO[idx];
                float term = value * h_x[col];
                sum += term;
                if (!first) cout << " + ";
                cout << value << " * " << h_x[col] << " (A[" << row << "][" << col << "] * x[" << col << "])";
                first = false;
            }
        }

        cout << " = " << sum << " (Host) vs " << h_y[row] << " (GPU)" << endl;
    }
}

// **CUDA Kernel for ELL Part**
__global__ void spmv_ell_kernel(int numRows, int numCols, int maxNonzerosPerRow,
                                unsigned int* colIdxELL, float* valueELL,
                                float* x, float* y) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < maxNonzerosPerRow; i++) {
            unsigned int idx = row * maxNonzerosPerRow + i;
            unsigned int col = colIdxELL[idx];
            if (col < numCols) {
                float value = valueELL[idx];
                sum += value * x[col];
            }
        }
        atomicAdd(&y[row], sum);
    }
}

// **CUDA Kernel for COO Part**
__global__ void spmv_coo_kernel(int numCOONonzeros,
                                unsigned int* rowIdxCOO, unsigned int* colIdxCOO, float* valueCOO,
                                float* x, float* y) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCOONonzeros) {
        unsigned int row = rowIdxCOO[idx];
        unsigned int col = colIdxCOO[idx];
        float value = valueCOO[idx];
        atomicAdd(&y[row], value * x[col]);
    }
}

// **Main Function**
int main() {
    // Define matrix dimensions
    int numRows = 4;
    int numCols = 4;
    int maxNonzerosPerRow = 2;
    int numELLRows = 4;
    int numCOONonzeros = 2;

    // Host arrays for the sample matrix
    // ELL part: first two nonzeros per row
    unsigned int h_colIdxELL[] = {0, 1, 0, 2, 1, 2, 2, 3};
    float h_valueELL[] = {1.0f, 7.0f, 5.0f, 3.0f, 2.0f, 1.0f, 8.0f, 6.0f};
    // COO part: additional nonzeros beyond maxNonzerosPerRow
    unsigned int h_rowIdxCOO[] = {1, 2};
    unsigned int h_colIdxCOO[] = {3, 3};
    float h_valueCOO[] = {4.0f, 9.0f};
    // Input vector x
    float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    // Output vector y
    float* h_y = new float[numRows]();

    // Device arrays
    unsigned int *d_colIdxELL, *d_rowIdxCOO, *d_colIdxCOO;
    float *d_valueELL, *d_valueCOO, *d_x, *d_y;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_colIdxELL, numELLRows * maxNonzerosPerRow * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_valueELL, numELLRows * maxNonzerosPerRow * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rowIdxCOO, numCOONonzeros * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_colIdxCOO, numCOONonzeros * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_valueCOO, numCOONonzeros * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, numCols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, numRows * sizeof(float)));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_colIdxELL, h_colIdxELL, numELLRows * maxNonzerosPerRow * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valueELL, h_valueELL, numELLRows * maxNonzerosPerRow * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowIdxCOO, h_rowIdxCOO, numCOONonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdxCOO, h_colIdxCOO, numCOONonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valueCOO, h_valueCOO, numCOONonzeros * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, numCols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, numRows * sizeof(float))); // Initialize y to zero

    // Set block size
    int blockSize = 256;

    // Compute grid sizes
    int ellGridSize = (numELLRows + blockSize - 1) / blockSize;
    int cooGridSize = (numCOONonzeros + blockSize - 1) / blockSize;

    // Print matrix in dense format
    printHybridMatrixDense(numRows, numCols, maxNonzerosPerRow, numELLRows,
                           h_colIdxELL, h_valueELL,
                           numCOONonzeros, h_rowIdxCOO, h_colIdxCOO, h_valueCOO);

    // Launch ELL kernel
    spmv_ell_kernel<<<ellGridSize, blockSize>>>(numELLRows, numCols, maxNonzerosPerRow,
                                                d_colIdxELL, d_valueELL,
                                                d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch COO kernel
    spmv_coo_kernel<<<cooGridSize, blockSize>>>(numCOONonzeros,
                                                d_rowIdxCOO, d_colIdxCOO, d_valueCOO,
                                                d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printArray(h_x, numCols, "Input Vector x");
    printCalculationProcess(numRows, numCols, maxNonzerosPerRow, numELLRows,
                            h_colIdxELL, h_valueELL,
                            numCOONonzeros, h_rowIdxCOO, h_colIdxCOO, h_valueCOO,
                            h_x, h_y);
    printArray(h_y, numRows, "Output Vector y");

    // Free device memory
    CUDA_CHECK(cudaFree(d_colIdxELL));
    CUDA_CHECK(cudaFree(d_valueELL));
    CUDA_CHECK(cudaFree(d_rowIdxCOO));
    CUDA_CHECK(cudaFree(d_colIdxCOO));
    CUDA_CHECK(cudaFree(d_valueCOO));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // Free host memory
    delete[] h_y;

    return 0;
}