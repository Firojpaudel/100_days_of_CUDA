/*Okay, so first; lets learn to build a CNN normally in host ie. no real use of CUDA but still let's try to implement it in code and next we will introduce parallelization*/

#include<iostream>
#include<vector>
#include<cmath>

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