#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "include/tensor.h"  // Include the Tensor class header
#include <random>
#include "include/linear_network.h"
#include "include/dataloader.h"
#include <fstream>
#include <sstream>

using namespace deepc;


int main() {
    Tensor<float> input1 = Tensor<float>({100,784},  std::vector<float>(100*784, 1.0f), true);
    
    int epoch = 100;
    float learning_rate = 0.01;

    std::vector<std::pair<Tensor<float>, Tensor<float>>> mnist_data = DataLoader::load_mnist("data/mnist_784.csv", 1000);

    for (int i = 0; i < epoch; i++) {

    }

    // First layer 100x784 * 784x16 = 100x16
    input1.getInfo();
    Tensor<float> l1_weight = Tensor<float>({784, 16}, std::vector<float>(784*16, 0.5), true);
    l1_weight.getInfo();
    Tensor<float> l1_bias   = Tensor<float>({1, 16}, std::vector<float>(1*16, 0.1), true);
    l1_bias.getInfo();
    Tensor<float> l1_output = input1.matmul2D(l1_weight);
    l1_output.getInfo();

    // Activation layer
    Tensor<float> relu1 = l1_output.relu();

    // 2nd layer  100x16 * 16x8 = 100x8
    Tensor<float> l2_weight = Tensor<float>({16, 8}, std::vector<float>(16*8, 0.5), true);
    l2_weight.getInfo();
    Tensor<float> l2_bias   = Tensor<float>({1, 8}, std::vector<float>(8*1, 0.1), true);
    l2_bias.getInfo();
    Tensor<float> l2_output = relu1.matmul2D(l2_weight);
    l2_output.getInfo();

    // Activation layer
    Tensor<float> relu2 = l2_output.relu();
    relu2.getInfo();
    
    // 3rd layer  100x8 * 8x4 = 100x4
    Tensor<float> l3_weight = Tensor<float>({8, 2}, std::vector<float>(8*2, 0.5), true);
    l3_weight.getInfo();
    Tensor<float> l3_bias   = Tensor<float>({1, 2}, std::vector<float>(1*2, 0.1), true);
    l3_bias.getInfo();
    Tensor<float> l3_output = relu2.matmul2D(l3_weight);
    l3_output.getInfo();
    Tensor<float> l3_bias_output = l3_output + l3_bias;
    l3_bias_output.getInfo();
    
    l3_bias_output.backward();

    LinearNetwork<float> nn;


}
