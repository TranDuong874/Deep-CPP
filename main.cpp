#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "include/tensor.h"  // Include the Tensor class header
#include <random>
#include "include/dataloader.h"
#include <fstream>
#include <sstream>
#include "include/sgd.h"
#include "include/linear_network.h"

using namespace deepc;

#include <fstream>
int main() {

    // Load the MNIST dataset
    std::vector<std::pair<std::vector<float>, std::vector<float>>> data = DataLoader::load_mnist("mnist_784.csv", 1000);

    int num_train   = 800;  
    int num_test    = 200;  

    std::vector<std::pair<std::vector<float>, std::vector<float>>> train_data(data.begin(), data.begin() + num_train);
    std::vector<std::pair<std::vector<float>, std::vector<float>>> test_data(data.begin() + num_train, data.end());

    std::vector<float> all_train_inputs;
    for (const auto& row : train_data) {
        all_train_inputs.insert(all_train_inputs.end(), row.first.begin(), row.first.end());
    }
    std::vector<float> all_train_targets;
    for (const auto& row : train_data) {
        all_train_targets.insert(all_train_targets.end(), row.second.begin(), row.second.end());
    }
    std::vector<float> all_test_inputs;
    for (const auto& row : test_data) {
        all_test_inputs.insert(all_test_inputs.end(), row.first.begin(), row.first.end());
    }
    std::vector<float> all_test_targets;
    for (const auto& row : test_data) {
        all_test_targets.insert(all_test_targets.end(), row.second.begin(), row.second.end());
    }

    Tensor<float>* train_inputs = new Tensor<float>({num_train, 784}, all_train_inputs, true);  
    Tensor<float>* train_targets = new Tensor<float>({num_train, 10}, all_train_targets, false);  
    Tensor<float>* test_inputs = new Tensor<float>({num_test, 784 }, all_test_inputs, true);  
    Tensor<float>* test_targets = new Tensor<float>({num_test, 10}, all_test_targets, false);  

    SGD* sgd = new SGD(0.9);  

    LinearNetwork nn(sgd);
    nn.addLayer(784, 32, "relu");
    nn.addLayer(32, 32, "relu");
    nn.addLayer(32, 10, "softmax");
    
    

    for (int epoch = 0; epoch < 100; epoch++) {  // Train for 10 epochs
        Tensor<float>* output = nn.forward(train_inputs);

        std::vector<float> output_values = output->getFlattenedVector();
        std::vector<float> target_values = train_targets->getFlattenedVector();
    
        // Print a few predictions vs true values during training
        std::cout << "Epoch " << epoch + 1 << " predictions:\n";
        for (int i = 0; i < std::min(10, num_train); ++i) {  // Print first 10 samples
            int pred_class = std::distance(output_values.begin() + i * 10, 
                                           std::max_element(output_values.begin() + i * 10, output_values.begin() + (i + 1) * 10));
    
            int true_class = std::distance(target_values.begin() + i * 10, 
                                           std::max_element(target_values.begin() + i * 10, target_values.begin() + (i + 1) * 10));
    
            std::cout << "Pred: " << pred_class << " True: " << true_class << std::endl;
        }

        // output->view();
    
        nn.backward(train_targets, "mse");
        std::cout << std::endl;
        // nn.getLayer(0)->weight->getGrad().view();
        std::cout << "Gradient norm: "; 
        nn.getLayer(0)->weight->getGrad().sum().view();
        std::cout << std::endl;

        nn.step();    
        nn.zeroGrad();  
    }

    std::cout << "Done" << std::endl;

    Tensor<float> test_pred = nn.predict(*test_inputs);
    // test_pred.view();

    std::vector<float> pred_values = test_pred.getFlattenedVector();
    std::vector<float> target_values = test_targets->getFlattenedVector();

    // Open a CSV file to write predictions
    std::ofstream csv_file("test_predictions.csv");
    csv_file << "Predicted,True\n";  // CSV header
    
    for (int i = 0; i < num_test; ++i) {
        int pred_class = std::distance(pred_values.begin() + i * 10, 
                                       std::max_element(pred_values.begin() + i * 10, pred_values.begin() + (i + 1) * 10));
    
        int true_class = std::distance(target_values.begin() + i * 10, 
                                       std::max_element(target_values.begin() + i * 10, target_values.begin() + (i + 1) * 10));
    
        csv_file << pred_class << "," << true_class << "\n";  // Write to CSV
    }
    
    csv_file.close();
    std::cout << "Predictions saved to test_predictions.csv" << std::endl;
    delete train_inputs;
    delete train_targets;
}
