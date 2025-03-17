#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;

// Structure for CSR (Compressed Sparse Row) graph representation
struct CSRGraph {
    unsigned int* srcPtrs;  // Pointers to start of each vertex's edges
    unsigned int* dst;      // Destination vertices array
    unsigned int numNodes;  // Total number of vertices
    unsigned int numEdges;  // Total number of edges
};

// Error handling macro
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while(0)

// CUDA kernel
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level,
                          unsigned int* prevFrontier, unsigned int* currFrontier,
                          unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
                          unsigned int currLevel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
             edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            /*In short about atomicCAS: Stands for atomic compare and swap: CUDA specific
            let's say we have: atomicCAS(address, compare, val) then this first:
            reads the current value at address, compares it with current value in the address;
            if these match: writes the val to address
             */
            if (atomicCAS(&level[neighbor], UINT_MAX, currLevel) == UINT_MAX) 
            /*So here; the operation checks if lavel[neighbor] is UINT_MAX (unvisited)
            If it is:
                - sets level[neighbor] to currLevel
                - returns UINT_MAX
            Else:
                - leaves level[neighbor] unchanged
                - returns the current value
            */
            {
                unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                currFrontier[currFrontierIdx] = neighbor;
            }
        }
    }
}

// Function to print the graph
void printGraph(CSRGraph& graph, vector<unsigned int>& h_srcPtrs, vector<unsigned int>& h_dst) {
    cout << "\nGraph Edges (CSR Format):\n";
    cout << "------------------------\n";
    cout << setw(10) << "Vertex" << setw(15) << "Neighbors" << endl;
    cout << "----------------------------\n";
    for (unsigned int i = 0; i < graph.numNodes; i++) {
        cout << setw(10) << i << setw(15);
        for (unsigned int j = h_srcPtrs[i]; j < h_srcPtrs[i + 1]; j++) {
            cout << h_dst[j] << " ";
        }
        cout << endl;
    }
}

// Function to print frontier contents
void printFrontier(const char* label, unsigned int* frontier, unsigned int size) {
    cout << setw(15) << label << ": ";
    if (size == 0) {
        cout << "Empty";
    } else {
        for (unsigned int i = 0; i < size; i++) {
            cout << frontier[i];
            if (i < size - 1) cout << ", ";
        }
    }
    cout << endl;
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
    const unsigned int numNodes = 5;
    const unsigned int numEdges = 6;
    
    // Host CSR data
    vector<unsigned int> h_srcPtrs = {0, 2, 4, 5, 6, 6};
    vector<unsigned int> h_dst = {1, 2, 2, 3, 4, 4};
    
    // Device pointers
    CSRGraph d_graph;
    unsigned int* d_level;
    unsigned int* d_prevFrontier;
    unsigned int* d_currFrontier;
    unsigned int* d_numCurrFrontier;
    
    // Host arrays for printing
    vector<unsigned int> h_prevFrontier(numNodes);
    vector<unsigned int> h_currFrontier(numNodes);
    
    // Allocate and copy source pointers
    CUDA_CHECK(cudaMalloc(&d_graph.srcPtrs, (numNodes + 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_graph.srcPtrs, h_srcPtrs.data(), 
                         (numNodes + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // Allocate and copy destination array
    CUDA_CHECK(cudaMalloc(&d_graph.dst, numEdges * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_graph.dst, h_dst.data(), 
                         numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    d_graph.numNodes = numNodes;
    d_graph.numEdges = numEdges;
    
    // Allocate and initialize level array
    CUDA_CHECK(cudaMalloc(&d_level, numNodes * sizeof(unsigned int)));
    unsigned int h_level[numNodes];
    for (unsigned int i = 0; i < numNodes; i++) {
        h_level[i] = UINT_MAX;
    }
    h_level[0] = 0;
    CUDA_CHECK(cudaMemcpy(d_level, h_level, numNodes * sizeof(unsigned int), 
                         cudaMemcpyHostToDevice));
    
    // Allocate frontier arrays
    CUDA_CHECK(cudaMalloc(&d_prevFrontier, numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_currFrontier, numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_numCurrFrontier, sizeof(unsigned int)));
    
    // Initialize first frontier
    h_prevFrontier[0] = 0;
    unsigned int h_numPrevFrontier = 1;
    CUDA_CHECK(cudaMemcpy(d_prevFrontier, h_prevFrontier.data(), 
                         sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // Print initial graph
    CSRGraph h_graph = {h_srcPtrs.data(), h_dst.data(), numNodes, numEdges};
    printGraph(h_graph, h_srcPtrs, h_dst);
    
    // BFS execution
    const int threadsPerBlock = 256;
    unsigned int currLevel = 1;
    unsigned int h_numCurrFrontier = 0;
    
    cout << "\nStarting BFS from vertex 0...\n";
    cout << "\nFrontier Progression:\n";
    cout << "--------------------\n";
    
    // Initial frontier print
    cout << "Level 0:\n";
    printFrontier("Prev Frontier", h_prevFrontier.data(), h_numPrevFrontier);
    printFrontier("Curr Frontier", h_currFrontier.data(), 0);
    cout << endl;
    
    // BFS loop
    while (h_numPrevFrontier > 0) {
        CUDA_CHECK(cudaMemset(d_numCurrFrontier, 0, sizeof(unsigned int)));
        
        int blocks = (h_numPrevFrontier + threadsPerBlock - 1) / threadsPerBlock;
        bfs_kernel<<<blocks, threadsPerBlock>>>(d_graph, d_level, 
                                              d_prevFrontier, d_currFrontier,
                                              h_numPrevFrontier, d_numCurrFrontier,
                                              currLevel);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy current frontier back to host
        CUDA_CHECK(cudaMemcpy(&h_numCurrFrontier, d_numCurrFrontier, 
                            sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_currFrontier.data(), d_currFrontier, 
                            h_numCurrFrontier * sizeof(unsigned int), 
                            cudaMemcpyDeviceToHost));
        
        // Print frontier status
        cout << "Level " << currLevel << ":\n";
        printFrontier("Prev Frontier", h_prevFrontier.data(), h_numPrevFrontier);
        printFrontier("Curr Frontier", h_currFrontier.data(), h_numCurrFrontier);
        cout << endl;
        
        // Swap frontiers
        CUDA_CHECK(cudaMemcpy(d_prevFrontier, d_currFrontier, 
                            h_numCurrFrontier * sizeof(unsigned int), 
                            cudaMemcpyDeviceToDevice));
        swap(h_prevFrontier, h_currFrontier);
        h_numPrevFrontier = h_numCurrFrontier;
        
        currLevel++;
    }
    
    // Copy final results
    CUDA_CHECK(cudaMemcpy(h_level, d_level, numNodes * sizeof(unsigned int), 
                         cudaMemcpyDeviceToHost));
    
    // Print results
    printLevels(h_level, numNodes);
    
    // Clean up
    CUDA_CHECK(cudaFree(d_graph.srcPtrs));
    CUDA_CHECK(cudaFree(d_graph.dst));
    CUDA_CHECK(cudaFree(d_level));
    CUDA_CHECK(cudaFree(d_prevFrontier));
    CUDA_CHECK(cudaFree(d_currFrontier));
    CUDA_CHECK(cudaFree(d_numCurrFrontier));
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}