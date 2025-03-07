#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <bitset>  
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Function to perform radix sort on a vector of integers (treating them as binary)
void radixSortBinary(vector<int>& arr) {
    // Find the maximum number to determine the number of bits needed
    int maxVal = *max_element(arr.begin(), arr.end());
    int numBits = 0;
    while (maxVal > 0) {
        maxVal >>= 1;  // Right shift to divide by 2
        numBits++;
    }

    // Do counting sort for every bit position
    for (int bitPos = 0; bitPos < numBits; bitPos++) {
        vector<int> output(arr.size());
        int i;
        vector<int> count(2, 0);

        // Count occurrences of 0 and 1 at the current bit position
        for (i = 0; i < arr.size(); i++) {
            count[(arr[i] >> bitPos) & 1]++; // Extract the bit at bitPos
        }

        // Modify count array to store the starting index in output for each bit value
        for (i = 1; i < 2; i++) {
            count[i] += count[i - 1];
        }

        // Build the output array
        for (i = arr.size() - 1; i >= 0; i--) {
            output[count[(arr[i] >> bitPos) & 1] - 1] = arr[i];
            count[(arr[i] >> bitPos) & 1]--;
        }

        // Copy the output array back to arr
        for (i = 0; i < arr.size(); i++) {
            arr[i] = output[i];
        }
    }
}

int main() {
    // Example data corresponding to the binary values in the image
    vector<int> data = {12, 3, 6, 9, 15, 8, 5, 10, 9, 6, 11, 13, 4, 10, 7, 0};

    cout << "Original array: ";
    for (int val : data) {
        cout << bitset<4>(val).to_string() << " "; // Display in binary format
    }
    cout << endl;

    // Measure execution time
    auto start = high_resolution_clock::now();
    radixSortBinary(data);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Sorted array: ";
    for (int val : data) {
        cout << bitset<4>(val).to_string() << " "; // Display in binary format
    }
    cout << endl;

    cout << "Time taken by Radix Sort: "
         << duration.count() << " microseconds" << endl;

    return 0;
}