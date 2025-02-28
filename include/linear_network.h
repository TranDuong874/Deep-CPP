#ifndef LINEAR_NETWORK_H
#define LINEAR_NETWORK_H

#include "tensor.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <fstream>
#include <sstream>
#include "layer.h"
#include "optimizer.h"

namespace deepc {
    class LinearNetwork {
        private:
            std::vector<Layer*>* layers;
            Tensor<float>* input;
            std::vector<Tensor<float>*> param_temp;
            std::vector<Tensor<float>*> loss_temp;

            Optimizer* opt;
            Tensor<float>* output;
        
        public:
            Layer* getLayer(int index) {
                return (*layers)[index];
            }

            LinearNetwork(Optimizer* opt) : opt(opt) {
                layers = new std::vector<Layer*>();
            }  
        
            void setOptimizer(Optimizer* opt) {
                this->opt = opt;
            }
        
            void step() {
                opt->step(this->layers); 
            }
        
            void addLayer(int input_features, int output_features, const std::string& activation = "") {
                layers->push_back(new Layer(input_features, output_features, activation));
            }
        
            Tensor<float>* forward(Tensor<float>* input_tensor) {
                input = input_tensor;
                output = nullptr;
                bool usedInput = false;
                for (auto layer : *layers) {
                    if (!usedInput) {
                        usedInput = true;
                        Tensor<float>* weighted = new Tensor<float>(this->input->matmul2D(*(layer->weight)));
                        param_temp.push_back(weighted);
                        Tensor<float>* biased = new Tensor<float>(*weighted + *(layer->bias));
                        param_temp.push_back(biased);
                    } else {
                        Tensor<float>* weighted = new Tensor<float>(param_temp.back()->matmul2D(*(layer->weight)));
                        param_temp.push_back(weighted);
                        Tensor<float>* biased = new Tensor<float>(*weighted + *(layer->bias));
                        param_temp.push_back(biased);
                    }
        
                    if (layer->activation == "sigmoid") {
                        Tensor<float>* activated = new Tensor<float>(param_temp.back()->sigmoid());
                        param_temp.push_back(activated);
                    } else if (layer->activation == "relu") {
                        Tensor<float>* activated = new Tensor<float>(param_temp.back()->relu());
                        param_temp.push_back(activated);
                    } else if (layer->activation == "tanh") {
                        Tensor<float>* activated = new Tensor<float>(param_temp.back()->tanh());
                        param_temp.push_back(activated);
                    } else if (layer->activation == "softmax") {
                        Tensor<float>* activated = new Tensor<float>(param_temp.back()->softmax2D());
                        param_temp.push_back(activated);
                    } else if (layer->activation == "leaky_relu") {
                        Tensor<float>* activated = new Tensor<float>(param_temp.back()->leaky_relu());
                        param_temp.push_back(activated);
                    }
                }
        
                this->output = param_temp.back();
                return this->output;
            }
        
            Tensor<float> predict(const Tensor<float>& input_tensor) {
                Tensor<float> output = input_tensor;
                output.setRequiresGrad(false);  // Disable gradient tracking
            
                for (auto& layer : *layers) {
                    output = output.matmul2D(*(layer->weight)) + *(layer->bias);
                    output.setRequiresGrad(false);  // Ensure each step is non-trainable
            
                    // Apply the activation function
                    if (layer->activation == "sigmoid") {
                        output = output.sigmoid();
                    } else if (layer->activation == "relu") {
                        output = output.relu();
                    } else if (layer->activation == "tanh") {
                        output = output.tanh();
                    } else if (layer->activation == "softmax") {
                        output = output.softmax2D();
                    } else if (layer->activation == "leaky_relu") {
                        output = output.leaky_relu();
                    }
                    output.setRequiresGrad(false);  // Final safety measure
                }
                zeroGrad();

                return output;
            }
            
            
        
            void backward(Tensor<float>* target, std::string loss="cross-entropy") {
                if (loss == "cross-entropy") {
                    Tensor<float>* log_output   = new Tensor<float>(this->output->log());  
                    Tensor<float>* multiplied   = new Tensor<float>(*target * *log_output);  
                    Tensor<float>* summed       = new Tensor<float>(multiplied->sum()); 
                    Tensor<float>* scalar       = new Tensor<float>({1,1}, {-1.0f}, true); 
                    Tensor<float>* neg          = new Tensor<float>(*summed * *scalar);
                    
                    Tensor<float>* scalar2      = new Tensor<float>(this->output->getShape()[0]);
                    Tensor<float>* loss         = new Tensor<float>(*neg / *scalar2);
                      
                    Tensor<float>* scalar3      = new Tensor<float>(this->output->getNumberOfElements());
                    Tensor<float>* averaged     = new Tensor<float>(*loss / *scalar3); 

                    loss_temp.push_back(log_output);
                    loss_temp.push_back(multiplied);
                    loss_temp.push_back(summed);
                    loss_temp.push_back(scalar);
                    loss_temp.push_back(neg);
                    loss_temp.push_back(loss);
                    loss_temp.push_back(averaged);

                    averaged->backward();
                   

                    std::cout << "cross entropy: "; 
                    averaged->view();
                    
                } else if (loss == "mse") {
                    Tensor<float>* diff         = new Tensor<float>(*output - *target);
                    
                    Tensor<float>* squared      = new Tensor<float>(diff->pow(2));
                    Tensor<float>* summed       = new Tensor<float>(squared->sum());
                    Tensor<float>* scalar       = new Tensor<float>(output->getNumberOfElements());
                    Tensor<float>* averaged     = new Tensor<float>(*summed / *scalar);

                    loss_temp.push_back(diff);
                    loss_temp.push_back(squared);
                    loss_temp.push_back(summed);
                    loss_temp.push_back(averaged);
                    loss_temp.push_back(scalar);

                    std::cout << "MSE: ";
                    averaged->view();

                    averaged->backward();
                }
            }
        
            void zeroGrad() {
                for (auto t : param_temp) {
                    delete t;
                }

                for (auto t : loss_temp) {
                    delete t;
                }

                param_temp.clear();
                loss_temp.clear();

                if (input) {
                    input->resetTensor();
                }
            
                for (auto layer : *layers) {
                    layer->weight->resetTensor();
                    layer->bias->resetTensor();
                }

                output = nullptr;
            }
        };
}

#endif 