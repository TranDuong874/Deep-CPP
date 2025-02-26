#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "include/tensor.h"  // Include the Tensor class header
#include <random>
#include "include/dataloader.h"
#include <fstream>
#include <sstream>

using namespace deepc;

struct Layer {
    Tensor<float>* weight;
    Tensor<float>* bias;
    std::string activation;

    Layer(int input_features, int output_features, const std::string& activation) {
        this->weight = new Tensor<float>({input_features, output_features}, std::vector<float>(input_features * output_features, 0.5f), true);
        this->bias = new Tensor<float>({1, output_features}, std::vector<float>(output_features, 0.0f), true);
        this->activation = activation;
    }
};

class Optimizer {
protected:
    float lr;  

public:
    Optimizer(float learning_rate) : lr(learning_rate) {}

    virtual void step(std::vector<Layer*>& layers) = 0; 
};

class SGD : public Optimizer {
public:
    SGD(float learning_rate) : Optimizer(learning_rate) {} 

    void step(std::vector<Layer*>& layers) override {
        for (auto& layer : layers) {
            if (layer->weight->requiresGrad()) {
                std::vector<float> tmp = layer->weight->getFlattenedVector();
                for (auto& x : tmp) {
                    x *= lr;
                }
                layer->weight->setValue(tmp);
            }
            if (layer->bias->requiresGrad()) {
                std::vector<float> tmp = layer->bias->getFlattenedVector();
                for (auto& x : tmp) {
                    x *= lr;
                }
                layer->bias->setValue(tmp);
            }
        }
    }
};

class LinearNetwork {
private:
    std::vector<Layer*> layers;
    Tensor<float>* input;
    std::vector<Tensor<float>*> temp;
    Optimizer* opt;

public:
    LinearNetwork(Optimizer* opt) : opt(opt) {}  

    void setOptimizer(Optimizer* opt) {
        this->opt = opt;
    }

    void step() {
        opt->step(this->layers); 
    }

    void addLayer(int input_features, int output_features, const std::string& activation = "") {
        layers.push_back(new Layer(input_features, output_features, activation));
    }

    Tensor<float>* forward(Tensor<float>* input_tensor) {
        input = input_tensor;

        bool usedInput = false;
        for (auto layer : layers) {
            if (!usedInput) {
                usedInput = true;
                Tensor<float>* weighted = new Tensor<float>(this->input->matmul2D(*(layer->weight)));
                temp.push_back(weighted);
                Tensor<float>* biased = new Tensor<float>(*weighted + *(layer->bias));
                temp.push_back(biased);
            } else {
                Tensor<float>* weighted = new Tensor<float>(temp.back()->matmul2D(*(layer->weight)));
                temp.push_back(weighted);
                Tensor<float>* biased = new Tensor<float>(*weighted + *(layer->bias));
                temp.push_back(biased);
            }

            if (layer->activation == "sigmoid") {
                Tensor<float>* activated = new Tensor<float>(temp.back()->sigmoid());
                temp.push_back(activated);
            } else if (layer->activation == "relu") {
                Tensor<float>* activated = new Tensor<float>(temp.back()->relu());
                temp.push_back(activated);
            } else if (layer->activation == "tanh") {
                Tensor<float>* activated = new Tensor<float>(temp.back()->tanh());
                temp.push_back(activated);
            }
        }

        return temp.back();
    }

    void backward() {
        temp.back()->backward();
    }

    void zeroGrad() {
        for (auto t : temp) {
            delete t;
        }
        temp.clear();
    
        if (input) {
            input->resetTensor();
        }
    
        for (auto layer : layers) {
            layer->weight->resetTensor();
            layer->bias->resetTensor();
        }
    }
};

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
    // Tensor<float>* test_inputs = new Tensor<float>({784, num_test}, all_test_inputs, true);  
    // Tensor<float>* test_targets = new Tensor<float>({10, num_test}, all_test_targets, false);  


    // Tensor<float>* input1 = new Tensor<float>({5, 2}, std::vector<float>(5 * 2, 0.1f), true);
    // Tensor<float>* test = new Tensor<float>({5, 2}, std::vector<float>(5 * 2, 0.3f), true);

    SGD* sgd = new SGD(0.01);  

    
    LinearNetwork nn(sgd);
    nn.addLayer(784, 128, "relu");
    nn.addLayer(128, 64, "relu");
    nn.addLayer(64, 10, "");

    for (int epoch = 0; epoch < 5; epoch++) {  // Train for 10 epochs
        Tensor<float>* output = nn.forward(train_inputs);
        output->view();
        
        // MSE
        Tensor<float>* diff = new Tensor<float>(*output - *train_targets);
        Tensor<float>* squared  = new Tensor<float>(diff->pow(2));
        Tensor<float>* summed = new Tensor<float>(squared->sum());
        Tensor<float>* averaged = new Tensor<float>(*summed / output->getNumberOfElements());
        

        averaged->view(); 
        std::cout << std::endl;

        averaged->backward();
        nn.step();    
        nn.zeroGrad();  

        delete diff;
        delete squared;
        delete summed;
        delete averaged;
        // delete output;
    }

    delete train_inputs;
    delete train_targets;
}
