#include <thrust/device_vector.h> 
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <iostream>

/*
    Thrust requires at least C++17 or higher. Might show flags(or might not run) when you have C++ lower than that.
*/

using namespace std;

// COO matrix structure
struct COOMatrix {
    float* values;
    int* row_indices;
    int* col_indices;
    int nnz;
    int num_rows;
};

// CSR matrix structure
struct CSRMatrix {
    float* values;
    int* col_indices;
    int* row_offsets;
    int nnz;
    int num_rows;
};

// Convert COO to CSR
void convertCOOtoCSR(const COOMatrix& coo, CSRMatrix& csr) {
    // Transfer COO data to device vectors
    thrust::device_vector<float> d_values(coo.values, coo.values + coo.nnz);
    thrust::device_vector<int> d_row_indices(coo.row_indices, coo.row_indices + coo.nnz);
    thrust::device_vector<int> d_col_indices(coo.col_indices, coo.col_indices + coo.nnz);

    // Step 1: Sort by row indices
    thrust::sort_by_key(d_row_indices.begin(), d_row_indices.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(d_col_indices.begin(), d_values.begin())));

    // Step 2: Compute row counts
    thrust::device_vector<int> row_counts(coo.num_rows);
    thrust::reduce_by_key(d_row_indices.begin(), d_row_indices.end(),
                          thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(),
                          row_counts.begin());

    // Step 3: Compute row offsets
    thrust::device_vector<int> row_offsets(coo.num_rows + 1);
    thrust::exclusive_scan(row_counts.begin(), row_counts.end(), row_offsets.begin());
    row_offsets[coo.num_rows] = coo.nnz; // Last element is nnz

    // Allocate and copy results to CSR
    csr.values = new float[coo.nnz];
    csr.col_indices = new int[coo.nnz];
    csr.row_offsets = new int[coo.num_rows + 1];
    thrust::copy(d_values.begin(), d_values.end(), csr.values);
    thrust::copy(d_col_indices.begin(), d_col_indices.end(), csr.col_indices);
    thrust::copy(row_offsets.begin(), row_offsets.end(), csr.row_offsets);
    csr.nnz = coo.nnz;
    csr.num_rows = coo.num_rows;
}

int main() {
    // COO matrix input (previous question input)
    COOMatrix coo;
    coo.nnz = 7;
    coo.num_rows = 4;
    coo.values = new float[coo.nnz]{1.0f, 7.0f, 8.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    coo.row_indices = new int[coo.nnz]{0, 0, 1, 2, 2, 3, 3};
    coo.col_indices = new int[coo.nnz]{0, 2, 2, 1, 2, 0, 3};

    // Convert to CSR
    CSRMatrix csr;
    convertCOOtoCSR(coo, csr);

    // Print results
    cout << "CSR format:\n";
    cout << "row_offsets: ";
    for (int i = 0; i <= csr.num_rows; i++) cout << csr.row_offsets[i] << " ";
    cout << "\ncol_indices: ";
    for (int i = 0; i < csr.nnz; i++) cout << csr.col_indices[i] << " ";
    cout << "\nvalues: ";
    for (int i = 0; i < csr.nnz; i++) cout << csr.values[i] << " ";
    cout << "\n";

    // Clean up
    delete[] coo.values;
    delete[] coo.row_indices;
    delete[] coo.col_indices;
    delete[] csr.values;
    delete[] csr.col_indices;
    delete[] csr.row_offsets;

    return 0;
}