#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <functional>
#include <algorithm>

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
        
            // From tensor index find the index in value_vector and the range of the data
            // e.g: tensor 2x2x3 if tensor[0] is chosen then get the values in value_vector that belongs the first 2x3 vector 
            std::pair<int,int> getFlatIndexAndStride(std::vector<int> index) {
                int flat_index = 0;
                int stride = value_vector.size();
                
                for (size_t nth_dim = 0; nth_dim < index.size(); nth_dim++) {
                    stride /= this->shape_vector[nth_dim];  
                    int offset = stride * index[nth_dim];
                    flat_index += offset;  
                }

                return std::pair<int, int>(flat_index, stride);
            }

            bool dimMatch(const Tensor<datatype>& other) {
                if (shape_vector.size() != other.shape_vector.size()) {
                    return false;
                }

                for (int i = 0; i < shape_vector.size(); i++) {
                    if (shape_vector[i] != other.shape_vector[i]) {
                        return false;
                    }
                }

                return true;
            }

            void initGrad(int data_length) {
                this->grad = std::vector<datatype>(data_length, 0.0);
            }


        public:
            // Create Tensor form another tensor
            Tensor(const Tensor<datatype>& other, bool requires_grad = false) {
                this->shape_vector = other.shape_vector;  
                this->value_vector = other.value_vector;
                setRequiresGrad(requires_grad);
            }
        
            // Create empty tensor with shape only
            // Data set to 0 by default
            Tensor(std::vector<int> shape_vector, bool requires_grad = false) {
                this->shape_vector = shape_vector;
                int data_length = 1;

                for (auto x : shape_vector) {
                    data_length *= x;
                }
                this->value_vector = std::vector<datatype>(data_length, 0.0f);
                setRequiresGrad(requires_grad);
            }

            // Create a full tensor with shape and value
            Tensor(std::vector<int> shape_vector, std::vector<datatype> value_vector, bool requires_grad = false) : Tensor(shape_vector) {
                setValue(value_vector);
                setRequiresGrad(requires_grad);
            }

            // Create a scalar tensor
            Tensor(datatype value, bool requires_grad = false) {
                this->shape_vector = {1};
                this->value_vector = {value};
                setRequiresGrad(requires_grad);
            }
            
            // Set data from a vector of data, the data will be automatically shaped into the vector current shape
            // If vector size doesnt match tensor size then return error message
            // Use this instead of this->value_vector = value_vector to ensure data matching
            void setValue(std::vector<datatype> value_vector) {
                if (value_vector.size() != this->value_vector.size()) {
                    throw std::runtime_error("Mismatch between value_vector length and tensor dimension");
                } else {
                    this->value_vector = value_vector;
                }
            }

            void setRequiresGrad(bool requires_grad) {
                initGrad(this->value_vector.size());
                this->requires_grad = requires_grad;
            }

            std::vector<int> getShape() {
                return this->shape_vector;
            }

            Tensor<datatype> getTensor(std::vector<int> index) {
                if (index.size() > this->shape_vector.size()) {
                    throw std::runtime_error("Too many indicies for tensor dimensions!");
                }   

                std::pair<int,int> index_stride = getFlatIndexAndStride(index);
                int flat_index = index_stride.first;
                int stride = index_stride.second;

                std::vector<int> new_shape;
                for (int i = index.size(); i < this->shape_vector.size(); i++) {
                    new_shape.push_back(this->shape_vector[i]);
                }

                if (new_shape.empty()) {
                    Tensor<datatype> result({1}); // Scalar tensor
                    setValue({this->value_vector[flat_index]});
                    return result;
                }

                Tensor<datatype> new_tensor(new_shape);
                
                std::vector<datatype> new_data;
                for (int i = flat_index; i < flat_index + stride; i++) {
                    new_data.push_back(this->value_vector[i]);
                }

                new_tensor.value_vector = new_data;

                return new_tensor;
            }

            void reshape(std::vector<int> shape_vector) {
                int data_size = 1;
                for (auto x : shape_vector) {
                    data_size *= x;
                }

                if (data_size != this->value_vector.size()) {
                    std::cout << "New shape_vector doesnt match length of value_vector" << std::endl;
                } else {
                    this->shape_vector = shape_vector;
                }
            }

            void view(int index = 0, int dim = 0, int indent = 0) {
                if (dim == shape_vector.size() - 1) {  // Last dimension: Print values on the same line
                    std::cout << std::string(indent, ' ') << "[";
                    for (int i = 0; i < shape_vector[dim]; i++) {
                        std::cout << value_vector[index + i];
                        if (i < shape_vector[dim] - 1) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << "]";
                    return;
                }
            
                std::cout << std::string(indent, ' ') << "[\n";
                int stride = 1;
                for (size_t i = dim + 1; i < shape_vector.size(); i++) {
                    stride *= shape_vector[i];
                }
            
                for (int i = 0; i < shape_vector[dim]; i++) {
                    view(index + i * stride, dim + 1, indent + 4);
                    if (i < shape_vector[dim] - 1) {
                        std::cout << ",\n";
                    }
                }
                std::cout << "\n" << std::string(indent, ' ') << "]\n";
            }

            Tensor<datatype> getGrad() const {
                return Tensor<datatype>(this->shape_vector, grad);
            }

            void backward() {
                if (!requires_grad) {
                    throw std::runtime_error("Called backward() on a non-requires_grad tensor");
                }

                if (this->grad.empty()) {
                    grad.resize(value_vector.size(), 1);
                } else {
                    std::fill(this->grad.begin(), grad.end(), 1);
                }            



                backward_recursion();
            }

            void backward_recursion() {
                if (grad_fn) {
                    grad_fn();

                    if (parent1) {
                        if (parent1->grad.empty()) {
                            parent1->grad.resize(parent1->value_vector.size(), 0);
                        }
                        parent1->backward_recursion();
                    }
                    if (parent2) {
                        if (parent2->grad.empty()) {
                            parent2->grad.resize(parent2->value_vector.size(), 0);
                        }
                        parent2->backward_recursion();
                    }
                }
            }

            Tensor<datatype> operator+(Tensor<datatype>& other) {
                std::pair<Tensor<datatype>, Tensor<datatype>> broadcastedPair = this->broadcast(other);
                Tensor<datatype> broadcasted_this = broadcastedPair.first;
                Tensor<datatype> broadcasted_other = broadcastedPair.second;
                
                Tensor<datatype> child(broadcasted_this.shape_vector, requires_grad || other.requires_grad);
            
                for (size_t i = 0; i < broadcasted_this.value_vector.size(); i++) {
                    child.value_vector[i] = broadcasted_this.value_vector[i] + broadcasted_other.value_vector[i];
                }
            
                if (requires_grad || other.requires_grad) {
                    child.parent1 = this;
                    child.parent2 = &other;
                    
                    child.grad_fn = [this, &other, &child, broadcasted_this, broadcasted_other]() {
                        std::vector<int> broadcast_dims;
                        
                        for (size_t i = 0; i < child.shape_vector.size(); i++) {
                            if (i >= this->shape_vector.size() || this->shape_vector[i] == 1) {
                                broadcast_dims.push_back(i);
                            }
                        }
 
                        if (this->requires_grad) {
                            Tensor<datatype> this_grad(child.shape_vector, false);
                            this_grad.value_vector = child.grad;
                        
                            if (!broadcast_dims.empty()) {
                                this_grad = this_grad.dimsum(broadcast_dims);
                            }
                            
                            for (size_t i = 0; i < this->grad.size(); i++) {
                                this->grad[i] += this_grad.value_vector[i];
                            }
                        }
                        
                        broadcast_dims.clear();
                        for (size_t i = 0; i < child.shape_vector.size(); i++) {
                            if (i >= other.shape_vector.size() || other.shape_vector[i] == 1) {
                                broadcast_dims.push_back(i);
                            }
                        }
                        
                        if (other.requires_grad) {
                            Tensor<datatype> other_grad(child.shape_vector, false);
                            other_grad.value_vector = child.grad;
                            
                            if (!broadcast_dims.empty()) {
                                other_grad = other_grad.dimsum(broadcast_dims);
                            }
                            
                            // Add to gradient
                            for (size_t i = 0; i < other.grad.size(); i++) {
                                other.grad[i] += other_grad.value_vector[i];
                            }
                        }
                    };
                }
                
                return child;
            }

            
            Tensor<datatype> operator-(Tensor<datatype>& other) {
                std::pair<Tensor<datatype>, Tensor<datatype>> broadcastedPair = this->broadcast(other);
                Tensor<datatype> broadcasted_this = broadcastedPair.first;
                Tensor<datatype> broadcasted_other = broadcastedPair.second;
                
                Tensor<datatype> child(broadcasted_this.shape_vector, requires_grad || other.requires_grad);
            
                for (size_t i = 0; i < broadcasted_this.value_vector.size(); i++) {
                    child.value_vector[i] = broadcasted_this.value_vector[i] - broadcasted_other.value_vector[i];
                }
            
                if (requires_grad || other.requires_grad) {
                    child.parent1 = this;
                    child.parent2 = &other;
                    
                    child.grad_fn = [this, &other, &child, broadcasted_this, broadcasted_other]() {
                        std::vector<int> broadcast_dims;
                        
                        for (size_t i = 0; i < child.shape_vector.size(); i++) {
                            if (i >= this->shape_vector.size() || this->shape_vector[i] == 1) {
                                broadcast_dims.push_back(i);
                            }
                        }
 
                        if (this->requires_grad) {
                            Tensor<datatype> this_grad(child.shape_vector, false);
                            this_grad.value_vector = child.grad;
                        
                            if (!broadcast_dims.empty()) {
                                this_grad = this_grad.dimsum(broadcast_dims);
                            }
                            
                            for (size_t i = 0; i < this->grad.size(); i++) {
                                this->grad[i] += this_grad.value_vector[i];
                            }
                        }
                        
                        broadcast_dims.clear();
                        for (size_t i = 0; i < child.shape_vector.size(); i++) {
                            if (i >= other.shape_vector.size() || other.shape_vector[i] == 1) {
                                broadcast_dims.push_back(i);
                            }
                        }
                        
                        if (other.requires_grad) {
                            Tensor<datatype> other_grad(child.shape_vector, false);
                            other_grad.value_vector = child.grad;
                            
                            if (!broadcast_dims.empty()) {
                                other_grad = other_grad.dimsum(broadcast_dims);
                            }
                            
                            // Add to gradient
                            for (size_t i = 0; i < other.grad.size(); i++) {
                                other.grad[i] -= other_grad.value_vector[i];
                            }
                        }
                    };
                }
                
                return child;
            }
            
            Tensor<datatype> operator*(Tensor<datatype>& other) {
                std::pair<Tensor<datatype>, Tensor<datatype>> broadcastedPair = this->broadcast(other);
                Tensor<datatype> broadcasted_this = broadcastedPair.first;
                Tensor<datatype> broadcasted_other = broadcastedPair.second;
                
                Tensor<datatype> child(broadcasted_this.shape_vector, requires_grad || other.requires_grad);
            
                for (size_t i = 0; i < broadcasted_this.value_vector.size(); i++) {
                    child.value_vector[i] = broadcasted_this.value_vector[i] * broadcasted_other.value_vector[i];
                }
            
                if (requires_grad || other.requires_grad) {
                    child.parent1 = this;
                    child.parent2 = &other;
                    
                    child.grad_fn = [this, &other, &child, broadcasted_this, broadcasted_other]() {
                        std::vector<int> broadcast_dims;
                        
                        for (size_t i = 0; i < child.shape_vector.size(); i++) {
                            if (i >= this->shape_vector.size() || this->shape_vector[i] == 1) {
                                broadcast_dims.push_back(i);
                            }
                        }
 
                        if (this->requires_grad) {
                            Tensor<datatype> this_grad(child.shape_vector, false);
                            this_grad.value_vector = child.grad;
                        
                            if (!broadcast_dims.empty()) {
                                this_grad = this_grad.dimsum(broadcast_dims);
                            }
                            
                            for (size_t i = 0; i < this->grad.size(); i++) {
                                this->grad[i] += this_grad.value_vector[i] * other.value_vector[i];
                            }
                        }
                        
                        broadcast_dims.clear();
                        for (size_t i = 0; i < child.shape_vector.size(); i++) {
                            if (i >= other.shape_vector.size() || other.shape_vector[i] == 1) {
                                broadcast_dims.push_back(i);
                            }
                        }
                        
                        if (other.requires_grad) {
                            Tensor<datatype> other_grad(child.shape_vector, false);
                            other_grad.value_vector = child.grad;
                            
                            if (!broadcast_dims.empty()) {
                                other_grad = other_grad.dimsum(broadcast_dims);
                            }
                            
                            // Add to gradient
                            for (size_t i = 0; i < other.grad.size(); i++) {
                                other.grad[i] += other_grad.value_vector[i] * this->value_vector[i];
                            }
                        }
                    };
                }
                
                return child;
            }

            Tensor<datatype> operator/(Tensor<datatype>& other) {
                std::pair<Tensor<datatype>, Tensor<datatype>> broadcastedPair = this->broadcast(other);
                Tensor<datatype> broadcasted_this = broadcastedPair.first;
                Tensor<datatype> broadcasted_other = broadcastedPair.second;
                
                Tensor<datatype> child(broadcasted_this.shape_vector, requires_grad || other.requires_grad);
            
                for (size_t i = 0; i < broadcasted_this.value_vector.size(); i++) {
                    child.value_vector[i] = broadcasted_this.value_vector[i] - broadcasted_other.value_vector[i];
                }
            
                if (requires_grad || other.requires_grad) {
                    child.parent1 = this;
                    child.parent2 = &other;
                    
                    child.grad_fn = [this, &other, &child, broadcasted_this, broadcasted_other]() {
                        std::vector<int> broadcast_dims;
                        
                        for (size_t i = 0; i < child.shape_vector.size(); i++) {
                            if (i >= this->shape_vector.size() || this->shape_vector[i] == 1) {
                                broadcast_dims.push_back(i);
                            }
                        }
 
                        if (this->requires_grad) {
                            Tensor<datatype> this_grad(child.shape_vector, false);
                            this_grad.value_vector = child.grad;
                        
                            if (!broadcast_dims.empty()) {
                                this_grad = this_grad.dimsum(broadcast_dims);
                            }
                            
                            for (size_t i = 0; i < this->grad.size(); i++) {
                                this->grad[i] += this_grad.value_vector[i] * 1/other.value_vector[i];;
                            }
                        }
                        
                        broadcast_dims.clear();
                        for (size_t i = 0; i < child.shape_vector.size(); i++) {
                            if (i >= other.shape_vector.size() || other.shape_vector[i] == 1) {
                                broadcast_dims.push_back(i);
                            }
                        }
                        
                        if (other.requires_grad) {
                            Tensor<datatype> other_grad(child.shape_vector, false);
                            other_grad.value_vector = child.grad;
                            
                            if (!broadcast_dims.empty()) {
                                other_grad = other_grad.dimsum(broadcast_dims);
                            }
                            
                            for (size_t i = 0; i < other.grad.size(); i++) {
                                other.grad[i] += other_grad.value_vector[i]  * (-this->value_vector[i] / (other.value_vector[i] * other.value_vector[i]));;
                            }
                        }
                    };
                }
                
                return child;
            }
             
            Tensor<datatype> matmul(Tensor<datatype> x, Tensor<datatype> y) {
                
            }
            
            Tensor<datatype> pow(int exponent) {
                Tensor<datatype> child(shape_vector, this->requires_grad);
            
                for (size_t i = 0; i < value_vector.size(); i++) {
                    child.value_vector[i] = std::pow(this->value_vector[i], exponent);
                }
            
                if (requires_grad) {
                    child.parent1 = this;
                    child.grad_fn = [this, &child, exponent]() {
                        for (size_t i = 0; i < child.grad.size(); i++) {
                            this->grad[i] += child.grad[i] * exponent * std::pow(this->value_vector[i], exponent - 1);
                        }
            
                    };
                }

            
                return child;
            }
            
            Tensor<datatype> sigmoid() {
                Tensor<datatype> child(shape_vector, requires_grad);
            
                for (size_t i = 0; i < value_vector.size(); i++) {
                    child.value_vector[i] = 1.0 / (1.0 + std::exp(-value_vector[i]));
                }
            
                if (requires_grad) {
                    child.parent1 = this;
                    child.grad_fn = [this, &child]() {
                        for (size_t i = 0; i < child.grad.size(); i++) {
                            datatype sig = child.value_vector[i];
                            this->grad[i] += child.grad[i] * sig * (1 - sig);
                        }
                    };
                }
            
                return child;
            }
            
            Tensor<datatype> tanh() {
                Tensor<datatype> child(shape_vector, requires_grad);
            
                for (size_t i = 0; i < value_vector.size(); i++) {
                    child.value_vector[i] = std::tanh(value_vector[i]);
                }
            
                if (requires_grad) {
                    child.parent1 = this;
                    child.grad_fn = [this, &child]() {
                        for (size_t i = 0; i < child.grad.size(); i++) {
                            datatype tanh_val = child.value_vector[i];
                            this->grad[i] += child.grad[i] * (1 - tanh_val * tanh_val);
                        }
                    };
                }
            
                return child;
            }

            Tensor<datatype> relu() {
                Tensor<datatype> child(shape_vector, requires_grad);
            
                // F(x) = max(0,x)
                for (size_t i = 0; i < value_vector.size(); i++) {
                    child.value_vector[i] = std::max(0.0, value_vector[i]);
                }
            
                if (requires_grad) {
                    child.parent1 = this;
            
                    child.grad_fn = [this, &child]() {
                        for (size_t i = 0; i < child.grad.size(); i++) {
                            if (this->value_vector[i] > 0) {
                                this->grad[i] += child.grad[i]; 
                            }
                        }
                    };
                }
            
                return child;
            }
            
            Tensor<datatype> exp() {
                Tensor<datatype> child(shape_vector, requires_grad);

                for (size_t i = 0; i < value_vector.size(); i++) {
                    child.value_vector[i] = std::exp(value_vector[i]);
                }

                if (requires_grad) {
                    child.parent1 = this;

                    child.grad_fn = [this, &child] {
                        for (size_t i = 0; i < child.grad.size(); i++) {
                            this->grad[i] += child.grad[i] * std::exp(this->value_vector[i]);
                        }
                    };
                }

                return child;
            }

            Tensor<datatype> sin() {
                Tensor<datatype> child(shape_vector, requires_grad);

                for (size_t i = 0; i < value_vector.size(); i++) {
                    child.value_vector[i] = std::sin(value_vector[i]);
                }

                if (requires_grad) {
                    child.parent1 = this;

                    child.grad_fn = [this, &child] {
                        for (size_t i = 0; i < child.grad.size(); i++) {
                            this->grad[i] += child.grad[i] * std::cos(this->value_vector[i]);
                        }
                    };
                }

                return child;
            }

            Tensor<datatype> cos() {
                
            }

            std::vector<datatype> getFlattenedVector() {
                return value_vector;
            }

            std::pair<Tensor<datatype>, Tensor<datatype>> broadcast(Tensor<datatype> &other) {
                std::vector<int> result_shape;
                std::vector<int> this_shape = this->shape_vector;
                std::vector<int> other_shape = other.shape_vector;

                // Pad 1 to the left of tensors with smaller size
                while (this_shape.size() < other_shape.size()) {
                    this_shape.insert(this_shape.begin(), 1);
                }
                while (other_shape.size() < this_shape.size()) {
                    other_shape.insert(other_shape.begin(), 1);
                }

                
                for (int i = 0; i < this_shape.size(); i++) {
                    if (this_shape[i] == other_shape[i]) {
                        result_shape.push_back(this_shape[i]);
                    } else if (this_shape[i] == 1) {
                        result_shape.push_back(other_shape[i]);
                    } else if (other_shape[i] == 1) {
                        result_shape.push_back(this_shape[i]);
                    } else {
                        throw std::runtime_error("Cannot broadcast tensors");
                    }
                }

                Tensor<datatype> this_broadcasted(result_shape, this->requires_grad);
                Tensor<datatype> other_broadcasted(result_shape, other.requires_grad);

                fillBroadcast(this_broadcasted, *this, this_shape);
                fillBroadcast(other_broadcasted, other, other_shape);

                // this_broadcasted.view();
                
                std::pair<Tensor<datatype>, Tensor<datatype>> broadcastedPair(this_broadcasted, other_broadcasted);
                return broadcastedPair;
            }      
            
            std::vector<int> flat_to_multi(size_t flat_index, const std::vector<int>& shape) const {
                int rank = shape.size();
                std::vector<int> multi_index(rank, 0);
                int temp_flat = flat_index;

                for (int j = rank - 1; j >= 0; j--) {
                    multi_index[j] = temp_flat % shape[j]; 
                    temp_flat /= shape[j];                 
                }
                return multi_index;
            }

            size_t multi_to_flat(const std::vector<int>& multi_index, const std::vector<int>& shape) const {
                int rank = shape.size();
                size_t flat_index = 0;
                size_t stride = 1;

                for (int j = rank - 1; j >= 0; j--) {
                    flat_index += multi_index[j] * stride;
                    stride *= shape[j];  
                }
                return flat_index;
            }

            Tensor<datatype> dimsum(const std::vector<int>& dims) {
                // Validate dimensions
                for (int dim : dims) {
                    if (dim < 0 || dim >= shape_vector.size()) {
                        throw std::runtime_error("Invalid dimension for summation");
                    }
                }
            
                // Sort dimensions in descending order to handle reduction properly
                std::vector<int> sorted_dims = dims;
                std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int>());
            
                // Create new shape by removing dimensions we're summing over
                std::vector<int> new_shape;
                for (int i = 0; i < shape_vector.size(); i++) {
                    if (std::find(sorted_dims.begin(), sorted_dims.end(), i) == sorted_dims.end()) {
                        new_shape.push_back(shape_vector[i]);
                    }
                }
                
                // If summing over all dimensions or resulting in a scalar
                bool is_scalar = new_shape.empty() || (new_shape.size() == 1 && new_shape[0] == 1);
                if (is_scalar) {
                    // Create a scalar tensor
                    Tensor<datatype> result({1}, requires_grad);
                    datatype sum = 0;
                    
                    // Sum all elements
                    for (const auto& val : value_vector) {
                        sum += val;
                    }
                    
                    result.value_vector[0] = sum;
                    
                    return result;
                }
            
                // Non-scalar case (same as before)
                Tensor<datatype> result(new_shape, requires_grad);
                std::fill(result.value_vector.begin(), result.value_vector.end(), 0);
            
                auto get_indices = [](int flat_idx, const std::vector<int>& shape) -> std::vector<int> {
                    std::vector<int> indices(shape.size());
                    for (int i = shape.size() - 1; i >= 0; i--) {
                        indices[i] = flat_idx % shape[i];
                        flat_idx /= shape[i];
                    }
                    return indices;
                };
            
                auto get_flat_index = [](const std::vector<int>& indices, const std::vector<int>& shape) -> int {
                    int flat_idx = 0;
                    int multiplier = 1;
                    for (int i = shape.size() - 1; i >= 0; i--) {
                        flat_idx += indices[i] * multiplier;
                        multiplier *= shape[i];
                    }
                    return flat_idx;
                };
            
                // Perform the summation
                for (size_t i = 0; i < value_vector.size(); i++) {
                    std::vector<int> src_indices = get_indices(i, shape_vector);
                    std::vector<int> dst_indices;
            
                    for (int j = 0; j < shape_vector.size(); j++) {
                        if (std::find(sorted_dims.begin(), sorted_dims.end(), j) == sorted_dims.end()) {
                            dst_indices.push_back(src_indices[j]);
                        }
                    }
            
                    int dst_idx = get_flat_index(dst_indices, new_shape);
                    result.value_vector[dst_idx] += value_vector[i];
                }

                return result;
            }

            void fillBroadcast(Tensor<datatype> &result_tensor, Tensor<datatype> &original_tensor, const std::vector<int>& padded_original_shape) {
                int rank = result_tensor.shape_vector.size();
                
                for (int i = 0; i < result_tensor.value_vector.size(); i++) {  
                    std::vector<int> tensor_index = result_tensor.flat_to_multi(i, result_tensor.shape_vector);
            
                    std::vector<int> original_index(rank, 0);
                    for (int j = 0; j < rank; j++) {
                        original_index[j] = (padded_original_shape[j] == 1) ? 0 : tensor_index[j];
                    }
            
                    int original_flat_index = original_tensor.multi_to_flat(original_index, padded_original_shape);
            
                    result_tensor.value_vector[i] = original_tensor.value_vector[original_flat_index];
                }
            }
            
            Tensor<datatype> matmul(Tensor<datatype> &other) {
                Tensor<datatype> broadcasted_pair = this->broadcast(other);
                Tensor<datatype> broadcasted_this = broadcasted_pair.first;
                Tensor<datatype> broadcasted_other = broadcasted_pair.second;
                
            }
            
    };                  
}                



//TO DO:
// 1. Add optimizers
// 2. Broadcasting capabilities
// 3. Unit testing for Tensor class
// 4. Refactor code
// 5. Check memory management
// 6. Add better error handling
// 7. Add broadcast for scalar * tensor operation
// 8. Matrix multiplication for 2D tensors
// 9. Rewrite reshape function, if new shape smaller than original, just trim, if new shape is greater, add zeros
// 10. Convolve function

int main() {

}