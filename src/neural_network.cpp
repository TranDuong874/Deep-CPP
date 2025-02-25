#include "../include/tensor.h"
#include "../include/linear.h"

namespace deepc {
    class NeuralNetwork {
        private:
            std::vector<LayerPtr> layers;

        public:
            NeuralNetwork() {};
            void addLayer(LayerPtr layer) {
                layers.push_back(layer);
            }

            void viewArchitecture() {
                int i = 0;
                for (auto layer : layers) {
                    std::cout << "Layer " << i << " ";
                    layer->getInfo();
                    i++;
                }
            }

            void forward(Tensor<float>& input) {
                Tensor<float> output = input;
                for (auto layer : layers) {
                    output = layer->forward(output);
                }
                
            }

            
    };
}

int main() {
    // Create a typical MNIST network
    deepc::Linear<float>* layer1 = new deepc::Linear<float>(784, 128); // Input layer to first hidden layer
    deepc::Linear<float>* layer2 = new deepc::Linear<float>(128, 64);  // First hidden layer to second hidden layer
    deepc::Linear<float>* layer3 = new deepc::Linear<float>(64, 10);   // Second hidden layer to output layer

    deepc::NeuralNetwork nn;

    nn.addLayer(deepc::LayerPtr(layer1));
    nn.addLayer(deepc::LayerPtr(layer2));
    nn.addLayer(deepc::LayerPtr(layer3));

    deepc::Tensor<float> input({1, 784}, std::vector<float>(784 * 1, 1.0), true);

    nn.forward(input);

    return 0;
}