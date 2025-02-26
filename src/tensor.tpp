#include "../include/tensor.h"

namespace deepc {
    template <class datatype>
    Tensor<datatype>::Tensor() {

    }
    
    // From tensor index find the index in value_vector and the range of the data
    // e.g: tensor 2x2x3 if tensor[0] is chosen then get the values in value_vector that belongs the first 2x3 vector 
    template <class datatype>
    std::pair<int,int> Tensor<datatype>::getFlatIndexAndStride(std::vector<int> index) {
        int flat_index = 0;
        int stride = value_vector.size();
        
        for (size_t nth_dim = 0; nth_dim < index.size(); nth_dim++) {
            stride /= this->shape_vector[nth_dim];  
            int offset = stride * index[nth_dim];
            flat_index += offset;  
        }

        return std::pair<int, int>(flat_index, stride);
    }

    template <class datatype>
    bool Tensor<datatype>::dimMatch(const Tensor<datatype>& other) {
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

    template <class datatype>
    void Tensor<datatype>::initGrad(int data_length) {
        this->grad = std::vector<datatype>(data_length, 0.0);
    }

    template <class datatype>
    Tensor<datatype>::Tensor(const Tensor<datatype>& other) {
        this->shape_vector = other.shape_vector;
        this->value_vector = other.value_vector;
        this->requires_grad = other.requires_grad;  
    
        this->parent1 = other.parent1;
        this->parent2 = other.parent2;
        this->grad_fn = other.grad_fn;
        this->grad = other.grad;
    }

    // // Create Tensor form another tensor
    // template <class datatype>
    // Tensor<datatype>::Tensor(const Tensor<datatype>& other, bool requires_grad) {
    //     this->shape_vector = other.shape_vector;  
    //     this->value_vector = other.value_vector;
    //     setRequiresGrad(requires_grad);
    // }

    // Create empty tensor with shape only
    // Data set to 0 by default
    template <class datatype>
    Tensor<datatype>::Tensor(std::vector<int> shape_vector, bool requires_grad) {
        this->shape_vector = shape_vector;
        int data_length = 1;

        for (auto x : shape_vector) {
            data_length *= x;
        }
        this->value_vector = std::vector<datatype>(data_length, 0.0f);
        setRequiresGrad(requires_grad);
    }

    // Create a full tensor with shape and value
    template <class datatype>
    Tensor<datatype>::Tensor(std::vector<int> shape_vector, std::vector<datatype> value_vector, bool requires_grad) : Tensor(shape_vector) {
        setValue(value_vector);
        setRequiresGrad(requires_grad);
    }

    // Create a scalar tensor
    template <class datatype>
    Tensor<datatype>::Tensor(datatype value, bool requires_grad) {
        this->shape_vector = {1};
        this->value_vector = {value};
        setRequiresGrad(requires_grad);
    }
    
    
    // Set data from a vector of data, the data will be automatically shaped into the vector current shape
    // If vector size doesnt match tensor size then return error message
    // Use this instead of this->value_vector = value_vector to ensure data matching
    template <class datatype>
    void Tensor<datatype>::setValue(std::vector<datatype> &value_vector) {
        if (value_vector.size() != this->value_vector.size()) {
            throw std::runtime_error("Mismatch between value_vector length and tensor dimension");
        } else {
            this->value_vector = value_vector;
        }
    }

    template <class datatype>
    void Tensor<datatype>::setValue(std::vector<int> index, Tensor<datatype> &tensor) {
        if (index.size() > this->shape_vector.size()) {
            throw std::runtime_error("Too many indices for tensor dimensions!");
        }   
    
        std::vector<int> target_shape;
        for (int i = index.size(); i < this->shape_vector.size(); i++) {
            target_shape.push_back(this->shape_vector[i]);
        }
    
        std::pair<int, int> index_stride = getFlatIndexAndStride(index);
        int flat_index = index_stride.first;
        int stride = index_stride.second;
    
        if (target_shape.empty()) {
            if (tensor.value_vector.size() != 1) {
                throw std::runtime_error("Expected scalar value for assignment");
            }
            this->value_vector[flat_index] = tensor.value_vector[0];
            return;
        }
    
        if (tensor.shape_vector != target_shape) {
            throw std::runtime_error("Shape mismatch for tensor assignment");
        }
    
        for (int i = 0; i < stride; i++) {
            this->value_vector[flat_index + i] = tensor.value_vector[i];
        }
    }

    template <class datatype>
    void Tensor<datatype>::setValue(Tensor<datatype> &tensor) {
        if (!dimMatch(tensor)) {
            throw std::runtime_error("Shape mismatch for tensor assignment");
        }

        this->value_vector = tensor.value_vector;
    }  

    template <class datatype>
    void Tensor<datatype>::setRequiresGrad(bool requires_grad) {
        initGrad(this->value_vector.size());
        this->requires_grad = requires_grad;
    }

    template <class datatype>
    std::vector<int> Tensor<datatype>::getShape() {
        return this->shape_vector;
    }

    template <class datatype>
    Tensor<datatype> Tensor<datatype>::getTensor(std::vector<int> index) {
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

    template <class datatype>
    void Tensor<datatype>::reshape(std::vector<int>& shape_vector) {
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

    template <class datatype>
    void  Tensor<datatype>::view(int index, int dim, int indent) {
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

    template <class datatype>
    Tensor<datatype> Tensor<datatype>::getGrad() {
        return Tensor<datatype>(this->shape_vector, grad);
    }

    template <class datatype>
    void Tensor<datatype>::backward() {
        if (!requires_grad) {
            throw std::runtime_error("Called backward() on a non-requires_grad tensor");
        }

       

        if (this->grad.empty()) {
            grad.resize(value_vector.size(), 1);
        } else {
            std::fill(this->grad.begin(), this->grad.end(), 1);
        }            

        backward_recursion();
    }

    template <class datatype>
    void Tensor<datatype>::backward_recursion() {
        if (grad_fn) {
            grad_fn();
            // std::cerr << std::endl;
            // std::cerr << ">>>>>\t" << this << std::endl;
            // std::cerr << this << "->" << "parent 1 = " << parent1 << std::endl;
            // std::cerr << this << "->" << "parent 2 = " << parent2 << std::endl;


            if (parent1 != nullptr) {
                // std::cerr << "visiting parent 1" << std::endl;
                // std::cerr << this << "->" << "parent 1 = " << parent1 << std::endl;
                if (parent1->grad.empty()) {
                    parent1->grad.resize(parent1->value_vector.size(), 0);
                }
                parent1->backward_recursion();
            }
            if (parent2 != nullptr) {
                // std::cerr << "visiting parent 2" << std::endl;
                // std::cerr << this << "->" << "parent 2 = " << parent2 << std::endl;
                if (parent2->grad.empty()) {
                    parent2->grad.resize(parent2->value_vector.size(), 0);
                }
                parent2->backward_recursion();
            }
        }
    }

    template <class datatype>
    Tensor<datatype> Tensor<datatype>::operator+(Tensor<datatype>& other) {
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

    
    template <class datatype>
    Tensor<datatype> Tensor<datatype>::operator-(Tensor<datatype>& other) {
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
    
    template <class datatype>
    Tensor<datatype> Tensor<datatype>::operator*(Tensor<datatype>& other) {
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
                std::cout << "* GRAD FN\n";
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

    template <class datatype>
    Tensor<datatype> Tensor<datatype>::operator/(Tensor<datatype>& other) {
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
                std::cout << "/ GRAD FN\n";
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
        
    
    template <class datatype>
    Tensor<datatype> Tensor<datatype>::pow(int exponent) {
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
    
    template <class datatype>
    Tensor<datatype> Tensor<datatype>::sigmoid() {
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
    
    template <class datatype>
    Tensor<datatype> Tensor<datatype>::tanh() {
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

    template <class datatype>
    Tensor<datatype> Tensor<datatype>::relu() {
        Tensor<datatype> child(shape_vector, requires_grad);
    
        // F(x) = max(0,x)
        for (size_t i = 0; i < value_vector.size(); i++) {
            child.value_vector[i] = std::max((datatype)0.0, value_vector[i]);
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
    
    template <class datatype>
    Tensor<datatype> Tensor<datatype>::exp() {
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

    template <class datatype>
    Tensor<datatype> Tensor<datatype>::sin() {
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

    template <class datatype>
    std::vector<datatype> Tensor<datatype>::getFlattenedVector() {
        return value_vector;
    }

    template <class datatype>
    std::pair<Tensor<datatype>, Tensor<datatype>> Tensor<datatype>::broadcast(Tensor<datatype> &other) {
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
    
    template <class datatype>
    std::vector<int> Tensor<datatype>::flat_to_multi(size_t flat_index, const std::vector<int>& shape) const {
        int rank = shape.size();
        std::vector<int> multi_index(rank, 0);
        int temp_flat = flat_index;

        for (int j = rank - 1; j >= 0; j--) {
            multi_index[j] = temp_flat % shape[j]; 
            temp_flat /= shape[j];                 
        }
        return multi_index;
    }

    template <class datatype>
    size_t Tensor<datatype>::multi_to_flat(const std::vector<int>& multi_index, const std::vector<int>& shape) const {
        int rank = shape.size();
        size_t flat_index = 0;
        size_t stride = 1;

        for (int j = rank - 1; j >= 0; j--) {
            flat_index += multi_index[j] * stride;
            stride *= shape[j];  
        }
        return flat_index;
    }

    template <class datatype>
    Tensor<datatype> Tensor<datatype>::dimsum(const std::vector<int>& dims) {
        for (int dim : dims) {
            if (dim < 0 || dim >= shape_vector.size()) {
                throw std::runtime_error("Invalid dimension for summation");
            }
        }
    
        // Sort dimensions in descending order to handle reduction properly
        std::vector<int> sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int>());
    
        // Create new shape by removing dimensions to sum over
        std::vector<int> new_shape;
        for (int i = 0; i < shape_vector.size(); i++) {
            if (std::find(sorted_dims.begin(), sorted_dims.end(), i) == sorted_dims.end()) {
                new_shape.push_back(shape_vector[i]);
            }
        }
        
        // Scalr case
        bool is_scalar = new_shape.empty() || (new_shape.size() == 1 && new_shape[0] == 1);
        if (is_scalar) {
            Tensor<datatype> result({1}, requires_grad);
            datatype sum = 0;
            
            for (const auto& val : value_vector) {
                sum += val;
            }
            
            result.value_vector[0] = sum;
            
            return result;
        }
    
        // Non-scalar case
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

    template <class datatype>
    Tensor<datatype> Tensor<datatype>::sum() {
        std::vector<int> result_shape = {1};
        Tensor<datatype> child(result_shape, this->requires_grad);
        
        datatype sum_value = 0;
        for (size_t i = 0; i < value_vector.size(); i++) {
            sum_value += this->value_vector[i];
        }
        
        child.value_vector[0] = sum_value;
        
        if (requires_grad) {
            child.parent1 = this;
            child.grad_fn = [this, &child]() {
                for (size_t i = 0; i < this->grad.size(); i++) {
                    this->grad[i] += child.grad[0]; 
                }
            };
        }
        
        return child;
    }

    template <class datatype>
    void Tensor<datatype>::fillBroadcast(Tensor<datatype> &result_tensor, Tensor<datatype> &original_tensor, const std::vector<int>& padded_original_shape) {
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

    template <class datatype>
    Tensor<datatype> Tensor<datatype>::matmul2D(Tensor<datatype> &other) {
        if (this->shape_vector.size() != 2 || other.shape_vector.size() != 2) {
            throw std::runtime_error("Matrix multiplication is only supported for 2D tensors");
        }
    
        int rowsA = this->shape_vector[0];
        int colsA = this->shape_vector[1];
        int rowsB = other.shape_vector[0];
        int colsB = other.shape_vector[1];
    
        if (colsA != rowsB) {
            throw std::runtime_error("Number of columns in the first matrix must equal the number of rows in the second matrix");
        }
    
        std::vector<int> result_shape = {rowsA, colsB};
        Tensor<datatype> result(result_shape, this->requires_grad || other.requires_grad);
        
        // Perform matrix multiplication
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result.value_vector[i * colsB + j] += this->value_vector[i * colsA + k] * other.value_vector[k * colsB + j];
                }
            }
        }
    
        if (result.requires_grad) {
            result.parent1 = this;
            result.parent2 = &other;
    
            
            result.grad_fn = [this, &other, &result, rowsA, colsA, colsB, rowsB]() {
                if (this->requires_grad) {
                    Tensor<datatype> this_grad(this->shape_vector, false);
                    std::fill(this_grad.value_vector.begin(), this_grad.value_vector.end(), 0);
                    
                    // dL/dX = dL/dZ * Y^T
                    for (int i = 0; i < rowsA; i++) {
                        for (int j = 0; j < colsA; j++) {
                            for (int k = 0; k < colsB; k++) {
                                this_grad.value_vector[i * colsA + j] += result.grad[i * colsB + k] * other.value_vector[j * colsB + k];
                            }
                        }
                    }
    
                    for (size_t i = 0; i < this->grad.size(); i++) {
                        this->grad[i] += this_grad.value_vector[i];
                    }
                }
    
                if (other.requires_grad) {
                    Tensor<datatype> other_grad(other.shape_vector, false);
                    std::fill(other_grad.value_vector.begin(), other_grad.value_vector.end(), 0);
                    
                    // dL/dY = X^T * dL/dZ
                    for (int i = 0; i < rowsB; i++) {
                        for (int j = 0; j < colsB; j++) {
                            for (int k = 0; k < rowsA; k++) {
                                other_grad.value_vector[i * colsB + j] += this->value_vector[k * colsA + i] * result.grad[k * colsB + j];
                            }
                        }
                    }
    
                    for (size_t i = 0; i < other.grad.size(); i++) {
                        other.grad[i] += other_grad.value_vector[i];
                    }
                }
            };
        }
    
        return result;
    }
    

    template <class datatype>
    Tensor<datatype> Tensor<datatype>::getFlattenTensor() {
        std::vector<int> new_shape = {1, (int)value_vector.size()};
        return Tensor<datatype>(new_shape, value_vector, requires_grad);
    }

    template <class datatype>
    Tensor<datatype> Tensor<datatype>::flatten() {
        this->shape_vector = {1, value_vector.size()};
    }              

    template <class datatype>
    int Tensor<datatype>::getNumberOfElements() {
        return this->value_vector.size();
    }

    template <class datatype>
    void Tensor<datatype>::getInfo() {
        std::cout << "Shape:\t";
        for (int i = 0; i < shape_vector.size(); i++) {
            std::cout << shape_vector[i] << " ";
        }
        std::cout << "\tRequires Grad: " << requires_grad;
        std::cout << "\tThis: " << this;
        std::cout << "\tParent1: " << parent1;
        std::cout << "\tParent2: " << parent2;
        std::cout << std::endl;
    }
    

    template <class datatype>
    bool Tensor<datatype>::requiresGrad() {
        return this->requires_grad;
    }
}