
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include "tensor.h"
#include <fstream>
#include <sstream>

#ifndef DATALOADER_H
#define DATALOADER_H

namespace deepc {
    class DataLoader {
        public:
            static std::vector<std::pair<Tensor<float>, Tensor<float>>> load_mnist(const std::string& filename, int num_samples) {
                std::ifstream file(filename);
                std::vector<std::pair<Tensor<float>, Tensor<float>>> dataset;
            
                if (!file.is_open()) {
                    throw std::runtime_error("Could not open file: " + filename);
                }
            
                std::string line;
            
                // Skip header row (if present)
                if (!std::getline(file, line)) {
                    throw std::runtime_error("File is empty or header missing.");
                }
            
                int count = 0;
                while (std::getline(file, line) && count < num_samples) {
                    std::stringstream ss(line);
                    std::vector<float> pixels;
                    std::string token;
                    int col = 0;
                    float label = 0.0f;
            
                    while (std::getline(ss, token, ',')) {
                        if (col == 0) {
                            // First token is the label
                            label = std::stof(token);
                        } else {
                            // Normalize pixel values
                            pixels.push_back(std::stof(token) / 255.0f);
                        }
                        col++;
                    }
            
                    // Ensure the input tensor has correct shape {1, 784}
                    Tensor<float> input({1, 784}, pixels, true);
            
                    // Convert label to one-hot encoding
                    std::vector<float> one_hot(10, 0.0f);
                    one_hot[static_cast<int>(label)] = 1.0f;
                    Tensor<float> target({1, 10}, one_hot, false);
            
                    dataset.emplace_back(input, target);
                    count++;
                }
            
                return dataset;
            }
    };
}

#endif