#ifndef LINEAR_H
#define LINEAR_H

#include "layer.h"
#include "tensor.h"

namespace deepc {
    template <class datatype>
    class Linear : public Layer {
        private:
            Tensor<datatype> weights;
            Tensor<datatype> bias;
            Tensor<datatype> input;
            Tensor<datatype> output;

        public:
            Linear(int in_features, int out_features) {
                std::vector<datatype> weight_vector = std::vector<datatype>(in_features * out_features, 0.5);
                std::vector<datatype> bias_vector = std::vector<datatype>(out_features, 0.0);

                weights = Tensor<datatype>({in_features, out_features}, weight_vector, true);
                bias = Tensor<datatype>({out_features}, bias_vector, true);
            }

            Linear(int in_features, int out_features, std::vector<datatype> weights, std::vector<datatype> bias) : Linear(in_features, out_features) {
                weights.setValue(weights);
                bias.setValue(bias);
            }

            void viewWeights() {
                weights.view();
            }

            void getInfo() {
                std::cout << "shape: " << weights.getShape()[0] << " x " << weights.getShape()[1] << std::endl;
            }

            virtual Tensor<float> forward(Tensor<float>& input) override {
                this->input = input;
                Tensor<datatype> output = input.matmul2D(weights) + bias;
                this->output = output;
                return output;
            }

            virtual void backward() override {
                
            }
    };
}

#endif