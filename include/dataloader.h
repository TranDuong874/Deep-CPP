#ifndef DATALOADER_H
#define DATALOADER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

namespace deepc {
    class DataLoader {
    public:
        static std::vector<std::pair<std::vector<float>, std::vector<float>>> load_mnist(const std::string& filename, int num_samples) {
            std::ifstream file(filename);
            std::vector<std::pair<std::vector<float>, std::vector<float>>> dataset;

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
                int label = 0; 

                while (std::getline(ss, token, ',')) {
                    if (col == 784) {
                       
                        label = std::stoi(token);  // Parse as integer
                    } else {
                        
                        pixels.push_back(std::stof(token) / 255.0f);
                    }
                    col++;
                }

                
                if (pixels.size() != 784) {
                    throw std::runtime_error("Invalid number of pixels in a row: expected 784, got " + std::to_string(pixels.size()));
                }

                
                std::vector<float> one_hot(10, 0.0f);
                if (label >= 0 && label < 10) {
                    one_hot[label] = 1.0f; 
                } else {
                    throw std::runtime_error("Invalid label value: " + std::to_string(label));
                }

                
                dataset.emplace_back(pixels, one_hot);
                count++;
            }

            return dataset;
        }
    };
}

#endif
