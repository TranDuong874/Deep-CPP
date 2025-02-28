#ifndef SGD_H
#define SGD_H
#include <vector>
#include "optimizer.h"

namespace deepc {
    class SGD : public Optimizer {
    public:
        SGD(float learning_rate) : Optimizer(learning_rate) {}

        void step(std::vector<Layer*>* &layers) override {
            for (auto& layer : *layers) {
                if (layer->weight->requiresGrad()) {
                    std::vector<float> weight_values = layer->weight->getFlattenedVector();
                    std::vector<float> weight_grads = layer->weight->getGrad().getFlattenedVector(); 
                    
                    for (size_t i = 0; i < weight_values.size(); i++) {
                        weight_values[i] -= lr * weight_grads[i]; 
                    }
                    // std::cout << "WEIGHT BEFORE: " << std::endl;
                    // layer->weight->view();
                    layer->weight->setValue(weight_values);
                    // layer->weight->getGrad().view();
                    // std::cout << "WEIGHT AFTER: " << std::endl;
                    // layer->weight->view();
                }
                
                if (layer->bias->requiresGrad()) {
                    std::vector<float> bias_values = layer->bias->getFlattenedVector();
                    std::vector<float> bias_grads = layer->bias->getGrad().getFlattenedVector(); 


                    for (size_t i = 0; i < bias_values.size(); i++) {
                        bias_values[i] -= lr * bias_grads[i]; 
                    }

                    layer->bias->setValue(bias_values);
                }
            }
        }
    };
}

#endif
