#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;

#define LOCAL_FRONTIER_CAPACITY 128

// Structure for CSR graph representation
struct CSRGraph {
    unsigned int* srcPtrs;  // Pointers to start of each vertex's edges
    unsigned int* dst;      // Destination vertices array
    unsigned int numNodes;  // Total number of vertices
    unsigned int numEdges;  // Total number of edges
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

// CUDA kernel with shared memory optimization
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level,
                          unsigned int* prevFrontier, unsigned int* currFrontier,
                          unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
                          unsigned int currLevel) {
    // Initialize privatized frontier in shared memory
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();

    // Perform BFS
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
             edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (atomicCAS(&level[neighbor], UINT_MAX, currLevel) == UINT_MAX) {
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
                if (currFrontierIdx_s < LOCAL_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    numCurrFrontier_s = LOCAL_FRONTIER_CAPACITY;
                    unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();

    // Allocate space in global frontier
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();

    // Commit to global frontier
    for (unsigned int currFrontierIdx_s = threadIdx.x;
         currFrontierIdx_s < numCurrFrontier_s; currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx = currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

// Function to print the graph
void printGraph(CSRGraph& graph, vector<unsigned int>& h_srcPtrs, vector<unsigned int>& h_dst) {
    cout << "\nGraph Edges (CSR Format):\n";
    cout << "-------------------------\n";
    cout << setw(12) << "Vertex" << setw(20) << "Neighbors" << endl;
    cout << "-----------------------------------\n";
    for (unsigned int i = 0; i < graph.numNodes; i++) {
        cout << setw(12) << i << setw(20);
        for (unsigned int j = h_srcPtrs[i]; j < h_srcPtrs[i + 1]; j++) {
            cout << h_dst[j] << " ";
        }
        cout << endl;
    }
}

// Function to print frontier contents
void printFrontier(const char* label, unsigned int* frontier, unsigned int size) {
    cout << setw(20) << label << ": ";
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
    cout << "\nFinal BFS Levels:\n";
    cout << "-----------------\n";
    cout << setw(12) << "Vertex" << setw(20) << "Level" << endl;
    cout << "-----------------------------------\n";
    for (unsigned int i = 0; i < numNodes; i++) {
        if (level[i] == UINT_MAX) {
            cout << setw(12) << i << setw(20) << "Unreachable" << endl;
        } else {
            cout << setw(12) << i << setw(20) << level[i] << endl;
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
    
    cout << "\nBFS Execution Starting from Vertex 0:\n";
    cout << "-------------------------------------\n";
    
    // Initial frontier print
    cout << "Level 0:\n";
    printFrontier("Previous Frontier", h_prevFrontier.data(), h_numPrevFrontier);
    printFrontier("Current Frontier", h_currFrontier.data(), 0);
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
        printFrontier("Previous Frontier", h_prevFrontier.data(), h_numPrevFrontier);
        printFrontier("Current Frontier", h_currFrontier.data(), h_numCurrFrontier);
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
    
    // Print final results
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