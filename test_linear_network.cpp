#include <iostream>
#include <vector>
#include <cassert>
#include "include/tensor.h"
#include "include/linear_network.h"
#include "include/layer.h"
#include "include/optimizer.h"
#include "include/sgd.h"

const float TOLERANCE = 1e-5;

bool compare(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        return false;
    }
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::fabs(vec1[i] - vec2[i]) > TOLERANCE) {
            return false;
        }
    }
    return true;
}

void test_linear_network() {
    // Create a simple linear network with ReLU activation
    deepc::SGD opt(0.01);
    deepc::LinearNetwork nn(&opt);

    nn.addLayer(2, 2, "relu");
    nn.addLayer(2, 1);

    // Initialize weights and biases for reproducibility
    nn.getLayer(0)->setWeights({0.5, 0.5, 0.5, 0.5});
    nn.getLayer(0)->setBias({0, 0});
    nn.getLayer(1)->setWeights({0.5, 0.5});
    nn.getLayer(1)->setBias({0});

    // Create a simple input tensor
    deepc::Tensor<float> input({1, 2}, {1.0, 2.0}, true);

    // Perform forward pass
    deepc::Tensor<float>* output = nn.forward(&input);

    // Expected output after ReLU activation
    std::vector<float> expected_output = {1.5}; // Adjust based on actual ReLU implementation

    // Verify the output
    assert(compare(output->getFlattenedVector(), expected_output));

    // Create a target tensor for backward pass
    deepc::Tensor<float> target({1, 1}, {1.0}, true);

    // Perform backward pass
    nn.backward(&target, "mse");
    std::cout << std::endl;

    // Expected gradients for layer1 and layer2
    std::vector<float> expected_grad_layer1 = {0.25, 0.25, 0.5, 0.5}; // Adjust based on actual gradient calculation
    std::vector<float> expected_grad_layer2 = {0.75, 0.75}; // Adjust based on actual gradient calculation

    // nn.getLayer(0)->weight->getGrad().view();
    // nn.getLayer(1)->weight->getGrad().view();
    // output->getGrad().view();

    // output->view();

    assert(compare(nn.getLayer(0)->weight->getGrad().getFlattenedVector(), expected_grad_layer1));
    assert(compare(nn.getLayer(1)->weight->getGrad().getFlattenedVector(), expected_grad_layer2));

    // Update weights
    nn.step();

    // Expected updated weights
    std::vector<float> expected_updated_weights_layer1 = {0.4975, 0.4975, 0.495, 0.495}; // Adjust based on actual update
    std::vector<float> expected_updated_weights_layer2 = {0.4925, 0.4925}; // Adjust based on actual update


    // Verify the updated weights
    assert(compare(nn.getLayer(0)->weight->getFlattenedVector(), expected_updated_weights_layer1));
    assert(compare(nn.getLayer(1)->weight->getFlattenedVector(), expected_updated_weights_layer2));

    std::cout << "test_linear_network passed!" << std::endl;
}

int main() {
    test_linear_network();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}