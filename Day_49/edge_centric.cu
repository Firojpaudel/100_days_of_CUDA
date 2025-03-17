#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;



// Structure for COO (Coordinate List) graph representation
struct COOGraph {
    unsigned int* src;      // Source vertices array
    unsigned int* dst;      // Destination vertices array
    unsigned int numEdges;  // Total number of edges
    unsigned int numNodes;  // Total number of vertices
};
// Error handling macro
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;      \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while(0)

// CUDA kernel
__global__ void bfs_kernel(COOGraph cooGraph, unsigned int* level,
                          unsigned int* newVertexVisited, unsigned int currLevel) {
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < cooGraph.numEdges) {
        unsigned int vertex = cooGraph.src[edge];
        if (level[vertex] == currLevel - 1) {
            unsigned int neighbor = cooGraph.dst[edge];
            if (level[neighbor] == UINT_MAX) {
                level[neighbor] = currLevel;
                *newVertexVisited = 1;
            }
        }
    }
}

// Function to print the graph
void printGraph(COOGraph& graph) {
    cout << "\nGraph Edges:\n";
    cout << "------------\n";
    cout << setw(10) << "Source" << setw(15) << "Destination" << endl;
    cout << "----------------------------\n";
    for (unsigned int i = 0; i < graph.numEdges; i++) {
        cout << setw(10) << graph.src[i] << setw(15) << graph.dst[i] << endl;
    }
}

// Function to print BFS levels
void printLevels(unsigned int* level, unsigned int numNodes) {
    cout << "\nBFS Levels:\n";
    cout << "-----------\n";
    cout << setw(10) << "Vertex" << setw(15) << "Level" << endl;
    cout << "----------------------------\n";
    for (unsigned int i = 0; i < numNodes; i++) {
        if (level[i] == UINT_MAX) {
            cout << setw(10) << i << setw(15) << "Unreachable" << endl;
        } else {
            cout << setw(10) << i << setw(15) << level[i] << endl;
        }
    }
}

int main() {
    // Example graph setup
    const unsigned int numNodes = 5;
    const unsigned int numEdges = 6;
    
    // Host graph data
    vector<unsigned int> h_src = {0, 0, 1, 1, 2, 3};
    vector<unsigned int> h_dst = {1, 2, 2, 3, 4, 4};
    
    // Device pointers
    COOGraph d_graph;
    unsigned int* d_level;
    unsigned int* d_newVertexVisited;
    
    // Allocate and copy source array
    CUDA_CHECK(cudaMalloc(&d_graph.src, numEdges * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_graph.src, h_src.data(), numEdges * sizeof(unsigned int), 
                         cudaMemcpyHostToDevice));
    
    // Allocate and copy destination array
    CUDA_CHECK(cudaMalloc(&d_graph.dst, numEdges * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_graph.dst, h_dst.data(), numEdges * sizeof(unsigned int), 
                         cudaMemcpyHostToDevice));
    
    // Set graph properties
    d_graph.numEdges = numEdges;
    d_graph.numNodes = numNodes;
    
    // Allocate and initialize level array
    CUDA_CHECK(cudaMalloc(&d_level, numNodes * sizeof(unsigned int)));
    unsigned int h_level[numNodes];
    for (unsigned int i = 0; i < numNodes; i++) {
        h_level[i] = UINT_MAX;
    }
    h_level[0] = 0; // Start from vertex 0
    CUDA_CHECK(cudaMemcpy(d_level, h_level, numNodes * sizeof(unsigned int), 
                         cudaMemcpyHostToDevice));
    
    // Allocate newVertexVisited flag
    CUDA_CHECK(cudaMalloc(&d_newVertexVisited, sizeof(unsigned int)));
    
    // Print initial graph
    COOGraph h_graph = {h_src.data(), h_dst.data(), numEdges, numNodes};
    printGraph(h_graph);
    
    // BFS execution
    const int threadsPerBlock = 256;
    const int blocks = (numEdges + threadsPerBlock - 1) / threadsPerBlock;
    
    unsigned int currLevel = 1;
    unsigned int h_newVertexVisited = 1;
    
    cout << "\nStarting BFS from vertex 0...\n";
    
    // BFS loop
    while (h_newVertexVisited) {
        CUDA_CHECK(cudaMemset(d_newVertexVisited, 0, sizeof(unsigned int)));
        
        bfs_kernel<<<blocks, threadsPerBlock>>>(d_graph, d_level, 
                                              d_newVertexVisited, currLevel);
        
        CUDA_CHECK(cudaGetLastError()); // Check kernel launch errors
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_newVertexVisited, d_newVertexVisited, 
                            sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        currLevel++;
    }
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_level, d_level, numNodes * sizeof(unsigned int), 
                         cudaMemcpyDeviceToHost));
    
    // Print results
    printLevels(h_level, numNodes);
    
    // Clean up
    CUDA_CHECK(cudaFree(d_graph.src));
    CUDA_CHECK(cudaFree(d_graph.dst));
    CUDA_CHECK(cudaFree(d_level));
    CUDA_CHECK(cudaFree(d_newVertexVisited));
    
    // Reset CUDA device
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}