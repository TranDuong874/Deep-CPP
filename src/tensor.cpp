#include <iostream>
#include <vector>
#include <stdexcept>

namespace deepc {
    template <class datatype>
    class Tensor {
        private:
            std::vector<datatype> data;
            std::vector<int> shape;
        
            std::pair<int,int> getFlatIndexAndStride(std::vector<int> index) {
                int flat_index = 0;
                int stride = data.size();
            
                for (size_t nth_dim = 0; nth_dim < index.size(); nth_dim++) {
                    stride /= this->shape[nth_dim];  
                    int offset = stride * index[nth_dim];
                    flat_index += offset;  
                }

                return std::pair<int, int>(flat_index, stride);
            }

            bool dimMatch(const Tensor<datatype>& other) {
                if (shape.size() != other.shape.size()) {
                    return false;
                }

                for (int i = 0; i < shape.size(); i++) {
                    if (shape[i] != other.shape[i]) {
                        return false;
                    }
                }

                return true;
            }


        public:
            Tensor(const Tensor<datatype>& other) {
                this->shape = other.shape;  
                this->data = other.data;
            }
        
            Tensor(std::vector<int> shape_vector) {
                this->shape = shape_vector;
                int data_length = 1;

                for (auto x : shape) {
                    data_length *= x;
                }

                this->data = std::vector<datatype>(data_length);
            }

            Tensor(std::vector<int> shape_vector, std::vector<datatype> data) : Tensor(shape_vector) {
                setData(data);
            }

            Tensor(datatype value) {
                this->shape = {1};
                this->setData({value});
            }
            
            void setData(std::vector<datatype> data) {
                if (data.size() != this->data.size()) {
                    throw std::runtime_error("Mismatch between data length and tensor dimension");
                } else {
                    this->data = data;
                }
            }

            
            std::vector<int> getShape() {
                return this->shape;
            }

            Tensor<datatype> getTensor(std::vector<int> index) {
                if (index.size() > this->shape.size()) {
                    throw std::runtime_error("Too many indicies for tensor dimensions!");
                }

                std::pair<int,int> index_stride = getFlatIndexAndStride(index);
                int flat_index = index_stride.first;
                int stride = index_stride.second;

                std::vector<int> new_shape;
                for (int i = index.size(); i < this->shape.size(); i++) {
                    new_shape.push_back(this->shape[i]);
                }

                if (new_shape.empty()) {
                    Tensor<datatype> result({1}); // Scalar tensor
                    result.setData({this->data[flat_index]});
                    return result;
                }

                Tensor<datatype> new_tensor(new_shape);
                
                std::vector<datatype> new_data;
                for (int i = flat_index; i < flat_index + stride; i++) {
                    new_data.push_back(this->data[i]);
                }

                new_tensor.setData(new_data);

                return new_tensor;
            }

            void reshape(std::vector<int> shape_vector) {
                int data_size = 1;
                for (auto x : shape_vector) {
                    data_size *= x;
                }

                if (data_size != this->data.size()) {
                    std::cout << "New shape doesnt match length of data" << std::endl;
                } else {
                    this->shape = shape_vector;
                }
            }

            void view(int index = 0, int dim = 0, int indent = 0) {
                if (dim == shape.size() - 1) {  // Last dimension: Print values on the same line
                    std::cout << std::string(indent, ' ') << "[";
                    for (int i = 0; i < shape[dim]; i++) {
                        std::cout << data[index + i] << (i < shape[dim] - 1 ? " " : "");
                    }
                    std::cout << "]\n";
                    return;
                }
        
                std::cout << std::string(indent, ' ') << "[\n";
                int stride = 1;
                for (size_t i = dim + 1; i < shape.size(); i++) {
                    stride *= shape[i];
                }
        
                for (int i = 0; i < shape[dim]; i++) {
                    view(index + i * stride, dim + 1, indent + 4);
                }
                std::cout << std::string(indent, ' ') << "]\n";
            }
        
            Tensor<datatype> operator*(const int value) {
                std::vector<datatype> result(this->data.size());
            
                for (size_t i = 0; i < this->data.size(); i++) {
                    result[i] = this->data[i] * value;
                }
            
                return Tensor<datatype>(this->shape, result); 
            }

            Tensor<datatype> operator+(const int value) {
                std::vector<datatype> result(this->data.size());
            
                for (size_t i = 0; i < this->data.size(); i++) {
                    result[i] = this->data[i] + value;
                }
            
                return Tensor<datatype>(this->shape, result); 
            }

            Tensor<datatype> operator-(const int value) {
                std::vector<datatype> result(this->data.size());
            
                for (size_t i = 0; i < this->data.size(); i++) {
                    result[i] = this->data[i] - value;
                }
            
                return Tensor<datatype>(this->shape, result); 
            }

            Tensor<datatype> operator/(const int value) {
                std::vector<datatype> result(this->data.size());
            
                for (size_t i = 0; i < this->data.size(); i++) {
                    result[i] = this->data[i] / value;
                }
            
                return Tensor<datatype>(this->shape, result); 
            }

            Tensor<datatype> operator+(const Tensor<datatype>& other) {
                if (!this->dimMatch(other)) {
                    throw std::runtime_error("Mismatch dimensions for element-wise addition");
                }
            
                std::vector<datatype> result(this->data.size());
                for (size_t i = 0; i < this->data.size(); i++) {
                    result[i] = this->data[i] + other.data[i];
                }
            
                return Tensor<datatype>(this->shape, result);
            }
            
            Tensor<datatype> operator-(const Tensor<datatype>& other) {
                if (!this->dimMatch(other)) {
                    throw std::runtime_error("Mismatch dimensions for element-wise subtraction");
                }
            
                std::vector<datatype> result(this->data.size());
                for (size_t i = 0; i < this->data.size(); i++) {
                    result[i] = this->data[i] - other.data[i];
                }
            
                return Tensor<datatype>(this->shape, result);
            }
            
            Tensor<datatype> operator*(const Tensor<datatype>& other) {
                if (!this->dimMatch(other)) {
                    throw std::runtime_error("Mismatch dimensions for element-wise multiplication");
                }
            
                std::vector<datatype> result(this->data.size());
                for (size_t i = 0; i < this->data.size(); i++) {
                    result[i] = this->data[i] * other.data[i];
                }
            
                return Tensor<datatype>(this->shape, result);
            }            

            std::vector<datatype> getFlattenedVector() {
                return data;
            }
    };
}

int main() {
    deepc::Tensor<float> tensor = deepc::Tensor<float>({3,4,6,6});

    std::vector<float> ones = std::vector<float>({3*4*6*6}, 0.5f);

    tensor.setData(ones);
    
    tensor.view();
    deepc::Tensor<float> tmp = tensor * 10;
    (tmp * tensor).view();

}