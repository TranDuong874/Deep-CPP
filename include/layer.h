#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <random>
#include <vector>
#include <cmath>

namespace deepc {
    struct Layer {
        Tensor<float>* weight;
        Tensor<float>* bias;
        std::string activation;

        Layer(int input_features, int output_features, const std::string& activation, unsigned int seed = 42) {
            std::mt19937 gen(seed); 
            
            // Xavier Initialization
            float limit = std::sqrt(6.0f / (input_features + output_features));
            std::uniform_real_distribution<float> dis(-limit, limit);

            // Initialize weights
            std::vector<float> weight_values(input_features * output_features);
            for (auto& val : weight_values) {
                val = dis(gen);
            }
            this->weight = new Tensor<float>({input_features, output_features}, weight_values, true);

            // Initialize biases to zero
            std::vector<float> bias_values(output_features, 0.0f);
            this->bias = new Tensor<float>({1, output_features}, bias_values, true);

            this->activation = activation;
        }

        void setWeights(std::vector<float> weights) {
            this->weight->setValue(weights);
        }

        void setBias(std::vector<float> bias) {
            this->bias->setValue(bias);
        }

        ~Layer() {
            delete weight;
            delete bias;
        }
    };
}

#endif
