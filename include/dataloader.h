#ifndef DATALOADER_H
#define DATALOADER_H

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
                float label = 0.0f;

                while (std::getline(ss, token, ',')) {
                    if (col == 0) {
                        // First token is the label
                        label = std::stof(token);
                    } else {
                        // Normalize pixel values to [0, 1]
                        pixels.push_back(std::stof(token) / 255.0f);
                    }
                    col++;
                }

                // Ensure the input has 784 pixels (for a 28x28 MNIST image)
                if (pixels.size() != 784) {
                    throw std::runtime_error("Invalid number of pixels in a row: expected 784, got " + std::to_string(pixels.size()));
                }

                // Convert label to one-hot encoding
                std::vector<float> one_hot(10, 0.0f);
                one_hot[static_cast<int>(label)] = 1.0f;

                // Add the data pair (input, target) to the dataset
                dataset.emplace_back(pixels, one_hot);
                count++;
            }

            return dataset;
        }
    };
}

#endif
