#include <iostream>
#include <cuda_runtime.h>
#include <iomanip> // For output formatting

using namespace std;

// Define CSR Matrix Structure
struct CSRMatrix {
    int numRows;          // Number of rows
    int numCols;          // Number of columns
    int numNonzeros;      // Number of nonzero elements
    unsigned int* rowPtrs; // Pointers to the start of each row
    unsigned int* colIdx;  // Column indices of non-zero elements
    float* value;         // Values of non-zero elements
};

// CUDA error check macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = call;                                               \
        if (error != cudaSuccess) {                                             \
            printf("CUDA error %04d: %s file: %s line: %d\n", error,            \
                   cudaGetErrorString(error), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Utility function to print an array on the host
void printArray(float* arr, int n, const char* label) {
    cout << "\n" << label << ":\n";
    cout << fixed << setprecision(2); // 2 decimal places for floats
    for (int i = 0; i < n; i++) {
        cout << "  [" << setw(2) << i << "] = " << setw(8) << arr[i];  //setw(2) -> setwidth(2) using this for clean output print formatting :) 
        if (i < n - 1) cout << "\n"; // Newline except for last element 😁
    }
    cout << endl;
}

// Function to print CSR matrix in dense format
void printCSRMatrixDense(int numRows, int numCols, int numNonzeros, unsigned int* rowPtrs, unsigned int* colIdx, float* value) {
    cout << "\nSparse Matrix A in dense format:\n";
    cout << fixed << setprecision(2); // Consistent float formatting
    for (int i = 0; i < numRows; i++) {
        int idx = rowPtrs[i]; // Start index for current row
        for (int j = 0; j < numCols; j++) {
            if (idx < rowPtrs[i + 1] && colIdx[idx] == j) {
                cout << setw(8) << value[idx];  
                idx++; // Move to next non-zero element in the row
            } else {
                cout << setw(8) << 0.00; // Print zero for empty positions
            }
        }
        cout << endl; // New line after each row
    }
}

// Function to print the calculation process for y = A * x
void printCalculationProcess(int numRows, int numNonzeros, unsigned int* rowPtrs, unsigned int* colIdx, float* value, float* x, float* y) {
    cout << "\nCalculation process for y = A * x:\n";
    cout << fixed << setprecision(2); // Consistent float formatting
    for (int i = 0; i < numRows; i++) {
        cout << "y[" << i << "] = ";
        bool first = true; // Flag for formatting terms
        float sum = 0.0f;  // Host-computed sum for this row
        for (int idx = rowPtrs[i]; idx < rowPtrs[i + 1]; idx++) {
            int j = colIdx[idx];
            float val = value[idx];
            float term = val * x[j];
            sum += term;
            if (!first) cout << " + ";
            cout << val << " * " << x[j] << " (A[" << i << "][" << j << "] * x[" << j << "])";
            first = false;
        }
        cout << " = " << sum << " (Host) vs " << y[i] << " (GPU)" << endl;
    }
}

// CUDA Kernel for SpMV with CSR (provided by you)
__global__ void spmv_csr_kernel(CSRMatrix csrMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < csrMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; ++i) {
            unsigned int col = csrMatrix.colIdx[i];
            float value = csrMatrix.value[i];
            sum += x[col] * value;
        }
        y[row] += sum;
    }
}

int main() {
    // Define matrix dimensions and number of non-zeros
    int numRows = 4;
    int numCols = 4;
    int numNonzeros = 8;

    // Dynamically allocate host arrays for CSR format
    unsigned int* h_rowPtrs = new unsigned int[numRows + 1]{0, 2, 5, 7, 8}; // One extra for the end
    unsigned int* h_colIdx = new unsigned int[numNonzeros]{0, 1, 0, 2, 3, 1, 2, 3};
    float* h_value = new float[numNonzeros]{1.0f, 7.0f, 5.0f, 3.0f, 9.0f, 2.0f, 8.0f, 6.0f};
    float* h_x = new float[numCols]{1.0f, 2.0f, 3.0f, 4.0f};  // Input vector
    float* h_y = new float[numRows]{0.0f, 0.0f, 0.0f, 0.0f};  // Output vector

    // Step 1: Display the sparse matrix A
    printCSRMatrixDense(numRows, numCols, numNonzeros, h_rowPtrs, h_colIdx, h_value);

    // Device arrays
    unsigned int *d_rowPtrs, *d_colIdx;
    float *d_value, *d_x, *d_y;

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc(&d_rowPtrs, (numRows + 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, numNonzeros * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_value, numNonzeros * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, numCols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, numRows * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_rowPtrs, h_rowPtrs, (numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, h_colIdx, numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, h_value, numNonzeros * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, numCols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, numRows * sizeof(float), cudaMemcpyHostToDevice));

    // Create CSRMatrix object on the host
    CSRMatrix csrMatrix;
    csrMatrix.numRows = numRows;
    csrMatrix.numCols = numCols;
    csrMatrix.numNonzeros = numNonzeros;
    csrMatrix.rowPtrs = d_rowPtrs;
    csrMatrix.colIdx = d_colIdx;
    csrMatrix.value = d_value;

    // Set block and grid size
    int blockSize = 256; 
    int numBlocks = (numRows + blockSize - 1) / blockSize; // Based on rows, not non-zeros

    // Launch kernel
    spmv_csr_kernel<<<numBlocks, blockSize>>>(csrMatrix, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost));

    // Step 2: Display input vector, calculation process, and output
    printArray(h_x, numCols, "Input Vector x");
    printCalculationProcess(numRows, numNonzeros, h_rowPtrs, h_colIdx, h_value, h_x, h_y);
    printArray(h_y, numRows, "Output Vector y");

    // Free memory on device
    CUDA_CHECK(cudaFree(d_rowPtrs));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // Free memory on host
    delete[] h_rowPtrs;
    delete[] h_colIdx;
    delete[] h_value;
    delete[] h_x;
    delete[] h_y;

    return 0;
}