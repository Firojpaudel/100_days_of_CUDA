#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// CSR graph structure
struct CSRGraph {
    unsigned int numVertices;
    unsigned int *srcPtrs; // Start indices in dst for each vertex
    unsigned int *dst;     // Destination vertices
};

// BFS kernel
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int *level,
                           unsigned int *newVertexVisited, unsigned int currLevel)
{
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < csrGraph.numVertices)
    {
        // Process vertices at the previous level
        if (level[vertex] == currLevel - 1)
        {
            for (unsigned int edge = csrGraph.srcPtrs[vertex];
                 edge < csrGraph.srcPtrs[vertex + 1]; ++edge)
            {
                unsigned int neighbor = csrGraph.dst[edge];
                if (level[neighbor] == UINT_MAX) // Unvisited neighbor
                {
                    level[neighbor] = currLevel;
                    atomicExch(newVertexVisited, 1U); // Atomically set flag
                }
            }
        }
    }
}

int main() {
    // Graph setup on host
    unsigned int numVertices = 6;
    vector<unsigned int> srcPtrs_host = {0, 2, 3, 4, 4, 4, 4}; // CSR pointers
    vector<unsigned int> dst_host = {1, 2, 3, 4};              // Edges
    // Graph: 0 -> 1,2; 1 -> 3; 2 -> 4; vertex 5 is isolated

    cout << "Starting BFS on graph with " << numVertices << " vertices." << endl;

    // Device memory allocation
    unsigned int *d_srcPtrs, *d_dst, *d_level, *d_newVertexVisited;
    CUDA_CHECK(cudaMalloc(&d_srcPtrs, (numVertices + 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_dst, dst_host.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_level, numVertices * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_newVertexVisited, sizeof(unsigned int)));

    // Copy graph data to device
    CUDA_CHECK(cudaMemcpy(d_srcPtrs, srcPtrs_host.data(),
                          (numVertices + 1) * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, dst_host.data(),
                          dst_host.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // Initialize level array: UINT_MAX except for start vertex
    CUDA_CHECK(cudaMemset(d_level, 0xFF, numVertices * sizeof(unsigned int)));
    unsigned int zero = 0;
    unsigned int start = 0; // Start from vertex 0
    CUDA_CHECK(cudaMemcpy(&d_level[start], &zero, sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // Configure CSRGraph for kernel
    CSRGraph csrGraph;
    csrGraph.numVertices = numVertices;
    csrGraph.srcPtrs = d_srcPtrs;
    csrGraph.dst = d_dst;

    // BFS loop
    unsigned int currLevel = 1;
    bool continueBFS = true;
    while (continueBFS) {
        cout << "Processing level " << currLevel << "..." << endl;
        CUDA_CHECK(cudaMemset(d_newVertexVisited, 0, sizeof(unsigned int)));

        // Launch kernel
        int blockSize = 256;
        int gridSize = (numVertices + blockSize - 1) / blockSize;
        bfs_kernel<<<gridSize, blockSize>>>(csrGraph, d_level,
                                            d_newVertexVisited, currLevel);
        CUDA_CHECK(cudaGetLastError()); // Check kernel launch errors
        CUDA_CHECK(cudaDeviceSynchronize()); // Check execution errors

        // Check if new vertices were visited
        unsigned int newVisited;
        CUDA_CHECK(cudaMemcpy(&newVisited, d_newVertexVisited,
                              sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (newVisited == 0) {
            cout << "No new vertices visited at level " << currLevel
                 << ", BFS complete." << endl;
            continueBFS = false;
        } else {
            cout << "New vertices visited, moving to level " << currLevel + 1
                 << "." << endl;
            currLevel++;
        }
    }

    // Copy results back to host
    vector<unsigned int> level_host(numVertices);
    CUDA_CHECK(cudaMemcpy(level_host.data(), d_level,
                          numVertices * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // Print BFS results
    cout << "\nBFS Results:" << endl;
    for (unsigned int i = 0; i < numVertices; ++i) {
        cout << "Vertex " << i << ": Level ";
        if (level_host[i] == UINT_MAX) {
            cout << "Unreachable" << endl;
        } else {
            cout << level_host[i] << endl;
        }
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_srcPtrs));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_level));
    CUDA_CHECK(cudaFree(d_newVertexVisited));

    return 0;
}