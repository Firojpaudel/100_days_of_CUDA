/*Okay, so first; lets learn to build a CNN normally in host ie. no real use of CUDA but still let's try to implement it in code and next we will introduce parallelization*/

#include<iostream>
#include<vector>
#include<cmath>
#include<chrono>
using namespace std;

/*Okay so before diving into code; let's first name the layers:
- Convolutional Layer
- Subsampling Layer
- Activation Function

Also, rn using just standard libs and avoiding external dependencies.
*/

// First let's define our activation function (Using sigmoid):
float sigmoid(float x){
    return 1.0f /(1.0f + exp(-x));
}

//Defining the MVP!
void convLayer_forward(
    const vector<vector<float>>& input,
    const vector<vector<float>>& filter,
    vector<vector<float>>& output,
    float bias
){
    int H = input.size();
    int W = input[0].size();
    int K = filter.size(); //Kernel 
    /*
    So before assigning the value to H_out and W_out, let's go through the general formula that we use to calculate the output dimensions of a convolution layer:
    
    O = { (W - K + 2P)/S } + 1 
    where,
    O == output size
    W == input size 
    K == Kernel size
    P == padding
    S == stride

    Now assuming the padding to be ZERO and stride of ONE; we can say the value of H_out and W_out are H - K +1 and W - K +1 respectively. (Since we are writing a simple CNN)
    */
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Initializing output with zeros
    output = vector<vector<float>>(H_out, vector<float>(W_out, 0.0f));

    //Sliding the kernel over the input
    for (int h = 0; h<H_out ; h++){
        for (int w = 0; w < W_out; w++){
            float sum = 0.0f;
            // Applying the kernel filter at position (h,w)
            for (int p= 0; p < K; p++){ //Rows
                for (int q=0; q< K; q++){ //Columns
                    sum += input[h+p][w+q] * filter[p][q];
                }
            }
            output[h][w] = sum + bias;
        }
    }
}

void subsamplingLayer_forward(
    const vector<vector<float>>& input, //Input Feature Map
    vector<vector<float>>& output, //Output after Pooling
    int K // Pooling Size of K x K
){
    int H = input.size();
    int W = input[0].size();
    /*
    Okay so here; taking some key assumptions:
    - Pooling window size: K x K
    - Stride = K (ie. window moves K pixels at a time)
    - No padding 
    - Input Dimensions are multiple of K (for clean division)

    Also dimensionality reduction is achieved so we will settle with H/K for now
    */
    int H_out = H/K;
    int W_out = W/K;

    // Initializing output with zeros
    output = vector<vector<float>>(H_out, vector<float>(W_out, 0.0f));

    //Average Pooling: Here we take average of each K x K region
    for (int h = 0; h < H_out; h++) {
        for (int w = 0; w < W_out; w++) {
            float sum = 0.0f;
            // Compute the average over the KxK region
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    sum += input[h * K + p][w * K + q];
                }
            }
            output[h][w] = sum / (K * K); // Average
        }
    }

    // Sigmoid Activation
    for (int h= 0; h < H_out; h++){
        for (int w= 0; w < W_out; w++){
            output[h][w] = sigmoid(output[h][w]);
        }
    }
}

void printMatrix(
    const vector<vector<float>> &matrix, 
    const string &name
){
    cout<<name<< ":\n";
    for (const auto& row : matrix) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}


int  main(){
    // Step 1: Defining a small input image (grayscale) 
    // I'll try to mimic A in a 5 x 5 grid 
    vector<vector<float>> input ={
        {0, 0, 1, 0, 0},
        {0, 1, 0, 1, 0},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1}, 
        {1, 0, 0, 0, 1}, 
    };
    printMatrix(input, "Input Image");

    //Step 2: Defining the kernel
    vector<vector<float>> filter ={
        {1, 0, 1},
        {0, 1, 0},
        {1, 0, 1} 
    };

    printMatrix(filter, "Filter");

    //Step 3: Applying the Convolution Layer
    vector<vector<float>> conv_output;
    float bias = 0.01f;
    
    //Step 4: Application of subsampling layer (2 x 2 average pooling):
    vector<vector<float>> pool_output;
    int pooling_size = 2;
    
    // Timing the convolution layer
    auto conv_start = chrono::high_resolution_clock::now();
    convLayer_forward(input, filter, conv_output, bias);
    auto conv_end = chrono::high_resolution_clock::now();
    auto conv_duration = chrono::duration_cast<chrono::microseconds>(conv_end - conv_start);
    double conv_time = conv_duration.count() / 1000.0; // Convert to milliseconds

    printMatrix(conv_output, "After Convolution");
    cout << "Convolution runtime: " << conv_time << " ms\n\n";

    // Timing the subsampling layer
    auto pool_start = chrono::high_resolution_clock::now();
    subsamplingLayer_forward(conv_output, pool_output, pooling_size);
    auto pool_end = chrono::high_resolution_clock::now();
    auto pool_duration = chrono::duration_cast<chrono::microseconds>(pool_end - pool_start);
    double pool_time = pool_duration.count() / 1000.0; // Convert to milliseconds

    printMatrix(pool_output, "After Pooling and Sigmoid");
    cout << "Subsampling runtime: " << pool_time << " ms\n";
    
    return 0;
}