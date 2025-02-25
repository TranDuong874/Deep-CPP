#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>

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
            Tensor();
            Tensor(const Tensor<datatype>& other, bool requires_grad = false);
            Tensor(std::vector<int> shape_vector, bool requires_grad = false);
            Tensor(std::vector<int> shape_vector, std::vector<datatype> value_vector, bool requires_grad = false);
            Tensor(datatype value, bool requires_grad = false);

            void setValue(std::vector<datatype>& value_vector);
            void setValue(std::vector<int> index, Tensor<datatype>& tensor);
            void setValue(Tensor<datatype>& tensor);
            void setRequiresGrad(bool requires_grad);
            int getNumberOfElements();
            std::vector<int> getShape();
            Tensor<datatype> getTensor(std::vector<int> index);
            void reshape(std::vector<int>& shape_vector);
            void view(int index = 0, int dim = 0, int indent = 0);
            Tensor<datatype> getGrad() const;
            void backward();
            void backward_recursion();

            Tensor<datatype> operator+(Tensor<datatype>& other);
            Tensor<datatype> operator-(Tensor<datatype>& other);
            Tensor<datatype> operator*(Tensor<datatype>& other);
            Tensor<datatype> operator/(Tensor<datatype>& other);

            Tensor<datatype> matmul2D(Tensor<datatype>& other);
            Tensor<datatype> pow(int exponent);
            Tensor<datatype> sigmoid();
            Tensor<datatype> tanh();
            Tensor<datatype> relu();
            Tensor<datatype> exp();
            Tensor<datatype> sin();
            Tensor<datatype> cos();

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