#include <iostream>
#include <vector>
#include <queue>
#include <cuda_runtime.h>

using namespace std;

// BFS function that takes the starting node and adjacency list
void bfs(int start, const vector<vector<int>>& adj) {
    int n = adj.size();               // Number of nodes (9 in this case)
    vector<int> distance(n, -1);      // Distance array, -1 means unvisited
    queue<int> q;                     // Queue for BFS traversal

    // Start with the root node
    q.push(start);
    distance[start] = 0;              // Distance to start node is 0

    // Process nodes until the queue is empty
    while (!q.empty()) {
        int current = q.front();      // Get the next node to process
        q.pop();
        cout << current << " ";       // Print the node (visit it)

        // Explore all neighbors of the current node
        for (int neighbor : adj[current]) {
            if (distance[neighbor] == -1) { // If neighbor hasn’t been visited
                q.push(neighbor);
                distance[neighbor] = distance[current] + 1; // Update distance
            }
        }
    }
    cout << endl; // Newline after traversal order

    // Print distances from the root to each node
    for (int i = 0; i < n; ++i) {
        cout << "Distance to " << i << ": " << distance[i] << endl;
    }
}

int main() {
    // Define the adjacency list based on the Mermaid graph
    vector<vector<int>> adj = {
        {1, 2},    // Node 0 -> 1, 2
        {3, 4},    // Node 1 -> 3, 4
        {5, 6, 7}, // Node 2 -> 5, 6, 7
        {8},       // Node 3 -> 8
        {8},       // Node 4 -> 8
        {},        // Node 5 -> (none)
        {8},       // Node 6 -> 8
        {},        // Node 7 -> (none)
        {}         // Node 8 -> (none)
    };

    bfs(0, adj);

    return 0;
}