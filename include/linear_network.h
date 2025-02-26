#ifndef LINEAR_NET_H
#define LINEAR_NET_H

#include "tensor.h"
#include <vector>
#include <string>
#include <stdexcept>

namespace deepc {
    template <class datatype>
    class LinearNetwork {
    private:
        struct Layer {
            Tensor<datatype> weights;
            Tensor<datatype> bias;
        };

        std::vector<Layer> layers;
        std::vector<Tensor<datatype>> intermediate;
        
    public:
        LinearNetwork() {}

        void addLayer(int input_features, int output_features) {
            layers.push_back(Layer());
            layers.back().weights = Tensor<datatype>(
                {input_features, output_features},
                std::vector<datatype>(input_features * output_features, 1),
                true
            );

            layers.back().bias = Tensor<datatype>(
                {1, output_features},
                std::vector<datatype>(output_features, 1),
                true
            );
        }

        Tensor<datatype> forward(Tensor<datatype> input_data) {
            if (layers.empty()) {
                throw std::runtime_error("Network has no layers.");
            }

            intermediate.clear();  // Reset intermediate storage
            intermediate.push_back(input_data);  // First input is stored

            intermediate.back().getInfo();
            
            for (size_t i = 0; i < layers.size(); i++) {
                std::cout << "Layer " << i << " --------------------------------------\n";
                std::cout << "Input:\n";
                intermediate.back().getInfo();

                std::cout << "Weight:\n";
                layers[i].weights.getInfo();

                intermediate.push_back(intermediate[intermediate.size() - 1].matmul2D(layers[i].weights));

                std::cout << "Weighted:\n";
                intermediate.back().getInfo();

                intermediate.push_back(intermediate[intermediate.size() - 1] + layers[i].bias);

                std::cout << "Bias:\n";
                layers[i].bias.getInfo();

                std::cout << "Output:\n";
                intermediate.back().getInfo();

            }

            return intermediate.back();  
        }

        void backward() {
            intermediate.back().backward();
        }
    };
}

#endif
