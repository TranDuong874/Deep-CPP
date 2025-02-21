#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <functional>

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

            // ChatGPT
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
                if (!dimMatch(other)) {
                    throw std::runtime_error("Shape mismatch in addition.");
                }

                Tensor<datatype> child(shape_vector, requires_grad || other.requires_grad);

                for (size_t i = 0; i < value_vector.size(); i++) {
                    child.value_vector[i] = value_vector[i] + other.value_vector[i];
                }

                if (child.requires_grad) {
                    child.parent1 = this;
                    child.parent2 = &other;

                    child.grad_fn = [this, &other, &child]() {
                        for (size_t i = 0; i < this->grad.size(); i++) {
                            if (this->requires_grad) this->grad[i] += 1.0f * child.grad[i];
                            if (other.requires_grad) other.grad[i] += 1.0f * child.grad[i];
                        }
                    };
                }

                return child;
            }
            
            Tensor<datatype> operator-(Tensor<datatype>& other) {
                if (!dimMatch(other)) {
                    throw std::runtime_error("Shape mismatch in subtraction.");
                }
            
                Tensor<datatype> child(shape_vector, requires_grad || other.requires_grad);
            
                for (size_t i = 0; i < value_vector.size(); i++) {
                    child.value_vector[i] = value_vector[i] - other.value_vector[i];
                }
            
                if (requires_grad || other.requires_grad) {
                    child.parent1 = this;
                    child.parent2 = &other;
                    child.grad_fn = [this, &other, &child]() {
                        for (size_t i = 0; i < child.grad.size(); i++) {
                            this->grad[i] += child.grad[i];  
                            other.grad[i] -= child.grad[i]; 
                        }
                    };
                }
            
                return child;
            }
            
            Tensor<datatype> operator*(Tensor<datatype>& other) {
                if (!dimMatch(other)) {
                    throw std::runtime_error("Shape mismatch in multiplication.");
                }
            
                Tensor<datatype> child(shape_vector, requires_grad || other.requires_grad);
            
                for (size_t i = 0; i < value_vector.size(); i++) {
                    child.value_vector[i] = value_vector[i] * other.value_vector[i];
                }
            
                if (requires_grad || other.requires_grad) {
                    child.parent1 = this;
                    child.parent2 = &other;
                    child.grad_fn = [this, &other, &child]() {
                        for (size_t i = 0; i < child.grad.size(); i++) {
                            this->grad[i] += child.grad[i] * other.value_vector[i];
                            other.grad[i] += child.grad[i] * this->value_vector[i];  
                        }
                    };
                }
            
                return child;
            }

            

            Tensor<datatype> operator/(Tensor<datatype>& other) {
                if (!dimMatch(other)) {
                    throw std::runtime_error("Shape mismatch in multiplication");
                }

                Tensor<datatype> child(shape_vector, requires_grad || other.requires_grad);

                for (size_t i = 0; i < value_vector.size(); i++) {
                    child.value_vector[i] = value_vector[i] / other.value_vector[i];
                }

                if (requires_grad || other.requires_grad) {
                    child.parent1 = this;
                    child.parent2 = &other;
                    child.grad_fn = [this, &other, child]() {
                        for (size_t i = 0; i < child.grad.size(); i++) {
                            this->grad[i] += child.grad[i] * 1/other.value_vector[i];
                            other.grad[i] += child.grad[i] * (-this->value_vector[i] / (other.value_vector[i] * other.value_vector[i]));
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

                this_broadcasted.view();
                
                std::pair<Tensor<datatype>, Tensor<datatype>> broadcastedPair(this_broadcasted, result_shape);
                return broadcastedPair;
            }           

            void fillBroadcast(Tensor<datatype> &result_tensor, Tensor<datatype> &original_tensor, std::vector<int> padded_original_shape) {
                int rank = result_tensor.shape_vector.size();
            
                for (int i = 0; i < result_tensor.value_vector.size(); i++) {  
                    int temp_flat = i;
                    std::vector<int> tensor_index(rank, 0);
            
                    // result flat -> result multi index
                    for (int j = rank - 1; j >= 0; j--) {
                        tensor_index[j] = temp_flat % result_tensor.shape_vector[j]; 
                        temp_flat /= result_tensor.shape_vector[j];                 
                    }
            
                    // Result tensor index -> original tensor index
                    std::vector<int> original_index(rank, 0);
                    for (int j = 0; j < rank; j++) {
                        if (padded_original_shape[j] == 1) {
                            original_index[j] = 0;
                        } else {
                            original_index[j] = tensor_index[j];
                        }  
                    }
            
                    // original multi -> original flat index
                    int original_flat_index = 0;
                    int stride = 1;
                    for (int j = rank - 1; j >= 0; j--) {
                        original_flat_index += original_index[j] * stride;
                        stride *= padded_original_shape[j];  
                    }
            
                    result_tensor.value_vector[i] = original_tensor.value_vector[original_flat_index];
                }
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
    // Init test
    deepc::Tensor<float> x({2,2}, {1.5,3.2,5.3,6.3}, true);
    x.view();

        // Init test
    deepc::Tensor<float> y({2,2}, {2.1,4.2,6.3,8.1}, true);
    y.view();

    // Operator test
    //f = x**2 + y**2 + x + y
    //df/dfx = 2x + 1
    deepc::Tensor<float> a = x.pow(2);
    deepc::Tensor<float> b = y.pow(2);
    deepc::Tensor<float> c = x + y;
    deepc::Tensor<float> d = a + b;
    deepc::Tensor<float> f = c + d;

    // Grad test
    f.backward();

    std::cout << "Gradient at x:\n";
    x.getGrad().view();

    std::cout << "Gradient at y:\n";
    y.getGrad().view();

    // Broadcasting test
    deepc::Tensor<float> z({3,2,1}, {3, 3, 3, 3, 3, 3}, false);

    x.broadcast(z);

    deepc::Tensor<float> h({1}, {3}, false);
}