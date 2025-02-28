#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip> 

namespace deepc {
    template <class datatype>
    class Tensor {
        private:
            std::vector<datatype> value_vector;
            std::vector<int> shape_vector;
            bool requires_grad = false;

            Tensor<datatype>* parent1 = nullptr;
            Tensor<datatype>* parent2 = nullptr;

            std::function<void()> grad_fn;
            std::vector<datatype> grad;
 
            std::pair<int, int> getFlatIndexAndStride(std::vector<int> index);
            bool dimMatch(const Tensor<datatype>& other);
            void initGrad(int data_length);

        public:
            ~Tensor() {
                resetTensor();
            }

            std::vector<datatype>& getValueVector() {
                return value_vector;
            }
    
            void resetTensor() {
                this->parent1 = nullptr;
                this->parent2 = nullptr;
                this->grad_fn = nullptr;

                setRequiresGrad(this->requires_grad);
                
                this->grad.clear();
                this->grad.resize(value_vector.size(), 0.0f);
            }

            void checkParents() {
                if (parent1 == nullptr) std::cout << "No parent1" << std::endl;
                else std::cout << parent2 << std::endl;
                if (parent2 == nullptr) std::cout << "No parent2" << std::endl;
                else std::cout << parent1 << std::endl;

            }
            Tensor();
            //Copy

            Tensor& operator=(const Tensor& other) {
                if (this != &other) {
                    this->shape_vector = other.shape_vector;
                    this->value_vector = other.value_vector;
                    this->requires_grad = other.requires_grad;
                    
                    this->parent1 = other.parent1;
                    this->parent2 = other.parent2;
                    this->grad_fn = other.grad_fn;
                    this->grad = other.grad;
                }

                return *this;
            }
            
            Tensor<datatype> sum();

            Tensor(const Tensor<datatype>& other);
            Tensor(std::vector<int> shape_vector, bool requires_grad = false);
            Tensor(std::vector<int> shape_vector, std::vector<datatype> value_vector, bool requires_grad = false);
            Tensor(datatype value, bool requires_grad = false);

            void setValue(std::vector<datatype>& value_vector);
            void setValue(std::vector<int> index, Tensor<datatype>& tensor);
            void setValue(Tensor<datatype>& tensor);
            void setRequiresGrad(bool requires_grad);
            bool requiresGrad();
            void getInfo();
            
            int getNumberOfElements();
            std::vector<int> getShape();
            Tensor<datatype> getTensor(std::vector<int> index);
            void reshape(std::vector<int>& shape_vector);
            void view(int index = 0, int dim = 0, int indent = 0);
            Tensor<datatype> getGrad();

            void backward();
            void backward_recursion();

            Tensor<datatype> operator*(float scalar) const {
                Tensor result = *this;
                for (auto& val : result.value_vector) {
                    val *= scalar;
                }
                return result;
            }

            Tensor<datatype> operator/(float scalar) const {
                Tensor result = *this;
                for (auto& val : result.value_vector) {
                    if (scalar == 0) throw std::runtime_error("Division of tensor by 0 scalar");
                    val /= scalar;
                }
                return result;
            }

            Tensor<datatype> operator+(Tensor<datatype>& other);
            Tensor<datatype> operator-(Tensor<datatype>& other);
            Tensor<datatype> operator*(Tensor<datatype>& other);
            Tensor<datatype> operator/(Tensor<datatype>& other);
            Tensor<datatype> operator=(Tensor<datatype>& other);

            Tensor<datatype> matmul2D(Tensor<datatype>& other);
            Tensor<datatype> pow(int exponent);
            Tensor<datatype> sigmoid();
            Tensor<datatype> tanh();
            Tensor<datatype> relu();
            Tensor<datatype> leaky_relu(float alpha = 0.01);
            Tensor<datatype> exp();
            Tensor<datatype> sin();
            Tensor<datatype> cos();
            Tensor<datatype> log();
            Tensor<datatype> softmax2D();

            std::vector<datatype> getFlattenedVector();
            std::pair<Tensor<datatype>, Tensor<datatype>> broadcast(Tensor<datatype>& other);
            std::vector<int> flat_to_multi(size_t flat_index, const std::vector<int>& shape) const;
            size_t multi_to_flat(const std::vector<int>& multi_index, const std::vector<int>& shape) const;
            Tensor<datatype> dimsum(const std::vector<int>& dims);
            void fillBroadcast(Tensor<datatype>& result_tensor, Tensor<datatype>& original_tensor, const std::vector<int>& padded_original_shape);
            Tensor<datatype> getFlattenTensor();
            Tensor<datatype> flatten();
    };
}

#include "../src/tensor.tpp"

#endif // TENSOR_H