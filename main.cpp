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
        this->bias = new Tensor<float>({1, output_features}, std::vector<float>(output_features, 1.5f), true);
        this->activation = activation;
    }
};

class LinearNetwork {
private:
    std::vector<Layer*> layers;
    Tensor<float>* input;
    std::vector<Tensor<float>*> temp;

public:
    void addLayer(int input_features, int output_features, const std::string& activation = "") {
        layers.push_back(new Layer(input_features, output_features, activation));
    }

    void forward(Tensor<float> input_tensor) {
        input = new Tensor<float>(input_tensor);

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
            
        }
    }

    void backward() {
        temp.back()->backward();
        layers.back()->weight->getGrad().view();
    }
};

int main() {
    Tensor<float> input1 = Tensor<float>({100,784},  std::vector<float>(100*784, 0.1f), true);

    LinearNetwork nn;
    nn.addLayer(784, 16);
    nn.addLayer(16, 2);

    nn.forward(input1);
    nn.backward();

}
