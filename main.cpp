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
    Tensor<float>* input1 = new Tensor<float>({5, 2}, std::vector<float>(5 * 2, 0.1f), true);
    Tensor<float>* test = new Tensor<float>({5, 2}, std::vector<float>(5 * 2, 0.3f), true);

    SGD* sgd = new SGD(0.01);  

    
    LinearNetwork nn(sgd);
    nn.addLayer(2, 16, "sigmoid");
    nn.addLayer(16, 2, "sigmoid");

    
    for (int epoch = 0; epoch < 10; epoch++) {  // Train for 10 epochs
        Tensor<float>* output = nn.forward(input1);
        Tensor<float>* loss = new Tensor<float>(*output - *test);

        
        loss->backward();
        nn.step();      // Apply optimizer
        nn.zeroGrad();  // Clear gradients

        delete loss;
        // delete output;
    }

    delete input1;
    delete test;
}
