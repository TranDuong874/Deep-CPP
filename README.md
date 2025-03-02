# C++ Neural Network from Scratch  

## Overview  
A simple artificial neural network (ANN) implemented in C++ from scratch. This project aims to provide a minimal yet functional implementation of a feedforward neural network with backpropagation.  

## Features  
- Fully connected feedforward architecture  
- Backpropagation for training  
- Support for multiple activation functions (ReLU, Sigmoid, etc.)  
- Configurable number of layers and neurons  
- Basic optimization methods. For now, only SGD is implemented  

### Prerequisites  
- C++17 or later  

# Use Guide  

## 1. Tensors  

The `Tensor` class serves as the foundation of the neural network, handling data storage and mathematical operations efficiently. 'Tensor' is basically a generalized case of matrix, you can think 'Tensor' as a matrix with more than 3 dimension. However, note that 3D matrices and smaller are also tensors.   

### **1.1 Creating a Tensor**  
You can create a tensor using different initialization methods:  

```cpp
#include "tensor.h"
using namespace deepc;

// Create an empty tensor
Tensor<float> t1;

// Create a tensor with specific shape (zeros by default)
Tensor<float> t2({2, 3}); // 2x3 tensor

// Create a tensor with specific shape and values
std::vector<float> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
Tensor<float> t3({2, 3}, values);

// Create a scalar tensor
Tensor<float> t4(5.0);
```

### **1.2. Basic Operations**
Tensors support various mathematic operations. Currently, 
```cpp
// Arithmetic operations
Tensor<float> a({2, 2}, {1, 2, 3, 4});
Tensor<float> b({2, 2}, {5, 6, 7, 8});

Tensor<float> sum = a + b;
Tensor<float> diff = a - b;
Tensor<float> product = a * b; // Element-wise multiplication
Tensor<float> quotient = a / b; // Element-wise division

// Scalar operations
Tensor<float> scaled = a * 2.5;
Tensor<float> divided = a / 2.0;

// Matrix multiplication
Tensor<float> matrixProduct = a.matmul2D(b);
```

### **1.3. Advanced Operations**
The library supports various mathematical functions and operations:
```cpp
// Element-wise operations
Tensor<float> powered = a.pow(2);
Tensor<float> activated = a.sigmoid();
Tensor<float> hyperbolic = a.tanh();
Tensor<float> rectified = a.relu();
Tensor<float> leakyRectified = a.leaky_relu(0.1);
Tensor<float> exponential = a.exp();
Tensor<float> logarithmic = a.log();
Tensor<float> trigSin = a.sin();
Tensor<float> trigCos = a.cos();

// Reduction operations
Tensor<float> summed = a.sum();

// Shape operations
std::vector<int> newShape = {1, 4};
a.reshape(newShape);
Tensor<float> flattened = a.flatten();
```

### **1.4. Gradients and Autograd**
The Tensor class supports automatic differentiation for backpropagation:
```cpp
// Create tensors that require gradients
Tensor<float> x({2, 2}, {1, 2, 3, 4}, true);
Tensor<float> y({2, 2}, {5, 6, 7, 8}, true);

// Perform operations
Tensor<float> z = x * y;
Tensor<float> loss = z.sum();

// Backpropagate gradients
loss.backward();

// Access gradients
Tensor<float> x_grad = x.getGrad();
Tensor<float> y_grad = y.getGrad();
```

### **1.5. Ultility Methods**
```cpp
// Get information about the tensor
tensor.getInfo(); // Prints shape and other details

// View the tensor contents
tensor.view(); // Prints tensor values

// Get tensor properties
std::vector<int> shape = tensor.getShape();
int elements = tensor.getNumberOfElements();
```
## 2. Neural Network
The neural network is built using the Tensor class and provides an interface for defining, training, and using feedforward neural networks.

### **2.1. Creating a Neural Network**
Currently, the codebase only support simple linear neural network with fully connected layers. CNN might be added in the future.
```cpp
#include "network.h"
using namespace deepc;

// Create a neural network with 3 layers (input: 2, hidden: 3, output: 1)
std::vector<int> architecture = {2, 3, 1};
Network<float> network(architecture);
```

### **2.2. Training the Network**
```cpp
// Prepare training data
Tensor<float> inputs({10, 2}, {...}); // 10 samples with 2 features each
Tensor<float> targets({10, 1}, {...}); // 10 target values

// Train for 1000 epochs with learning rate 0.01
network.train(inputs, targets, 1000, 0.01);
```

### **2.3. Making predictitons**
```cpp
// Create input tensor
Tensor<float> newData({1, 2}, {0.5, 0.8});

// Get prediction
Tensor<float> prediction = network.predict(newData);
```

## Implementation Details
### Tensor Class
The Tensor class is templated to support different data types and implements a computational graph for automatic differentiation. Key features include:
- Multi-dimensional data representation
- Broadcasting support for operations between tensors of different shapes
- Automatic gradient computation and backpropagation
- Extensive mathematical operations library
- Memory management with smart pointers to prevent leaks

### Neural Network Implementation
The neural network implementation features:
- Layer-wise architecture with configurable activation functions
- Stochastic Gradient Descent (SGD) optimizer
- Loss function implementations (MSE, Cross-Entropy)
- Forward and backward propagation for training
- Weight initialization strategies
