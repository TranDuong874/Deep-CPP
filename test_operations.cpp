#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "include/tensor.h"  // Include the Tensor class header

const float TOLERANCE = 1e-5;

bool compare(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        return false;
    }
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::fabs(vec1[i] - vec2[i]) > TOLERANCE) {
            return false;
        }
    }
    return true;
}

void test_grad_addition() {
    deepc::Tensor<float> x({2, 2}, {1.0, 2.0, 3.0, 4.0}, true);
    deepc::Tensor<float> y({2, 2}, {5.0, 6.0, 7.0, 8.0}, true);

    deepc::Tensor<float> z = x + y;
    z.backward();

    std::vector<float> expected_grad = {1.0, 1.0, 1.0, 1.0};
    
    assert(compare(x.getGrad().getFlattenedVector(), expected_grad));
    assert(compare(y.getGrad().getFlattenedVector(), expected_grad));

    std::cout << "test_grad_addition passed!" << std::endl;
}

void test_grad_multiplication() {
    deepc::Tensor<float> x({2, 2}, {1.0, 2.0, 3.0, 4.0}, true);
    deepc::Tensor<float> y({2, 2}, {5.0, 6.0, 7.0, 8.0}, true);

    deepc::Tensor<float> z = x * y;
    z.backward();

    std::vector<float> expected_grad_x = {5.0, 6.0, 7.0, 8.0};
    std::vector<float> expected_grad_y = {1.0, 2.0, 3.0, 4.0};
    assert(compare(x.getGrad().getFlattenedVector(), expected_grad_x));
    assert(compare(y.getGrad().getFlattenedVector(), expected_grad_y));

    std::cout << "test_grad_multiplication passed!" << std::endl;
}

void test_grad_power() {
    deepc::Tensor<float> x({2, 2}, {1.0, 2.0, 3.0, 4.0}, true);

    deepc::Tensor<float> z = x.pow(2);
    z.backward();

    std::vector<float> expected_grad = {2.0, 4.0, 6.0, 8.0};
    assert(compare(x.getGrad().getFlattenedVector(), expected_grad));

    std::cout << "test_grad_power passed!" << std::endl;
}

void test_grad_complex_expression() {
    deepc::Tensor<float> x({2, 2}, {1.0, 2.0, 3.0, 4.0}, true);
    deepc::Tensor<float> y({2, 2}, {5.0, 6.0, 7.0, 8.0}, true);

    // f = x^2 + y^2 + x + y
    deepc::Tensor<float> a = x.pow(2);
    deepc::Tensor<float> b = y.pow(2);
    deepc::Tensor<float> c = x + y;
    deepc::Tensor<float> d = a + b;
    deepc::Tensor<float> f = c + d;

    f.backward();

    std::vector<float> expected_grad_x = {2.0 * 1.0 + 1.0, 2.0 * 2.0 + 1.0, 2.0 * 3.0 + 1.0, 2.0 * 4.0 + 1.0};
    std::vector<float> expected_grad_y = {2.0 * 5.0 + 1.0, 2.0 * 6.0 + 1.0, 2.0 * 7.0 + 1.0, 2.0 * 8.0 + 1.0};
    assert(compare(x.getGrad().getFlattenedVector(), expected_grad_x));
    assert(compare(y.getGrad().getFlattenedVector(), expected_grad_y));

    std::cout << "test_grad_complex_expression passed!" << std::endl;
}

void test_grad_broadcasting() {
    deepc::Tensor<float> x({2, 2}, {1.0, 2.0, 3.0, 4.0}, true);
    deepc::Tensor<float> y({2, 1}, {5.0, 6.0}, true);

    // f = x + y (broadcasting y to match x's shape)
    deepc::Tensor<float> f = x + y;
    f.backward();

    std::vector<float> expected_grad_x = {1.0, 1.0, 1.0, 1.0};
    std::vector<float> expected_grad_y = {2.0, 2.0};  // y is broadcasted, so its gradient is summed over the broadcasted dimensions

    assert(compare(x.getGrad().getFlattenedVector(), expected_grad_x));
    assert(compare(y.getGrad().getFlattenedVector(), expected_grad_y));

    std::cout << "test_grad_broadcasting passed!" << std::endl;
}

void test_grad_matmul2D() {
    deepc::Tensor<float> x({2, 3}, 
        {
            1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0
        }, true);
    deepc::Tensor<float> y({3, 2}, 
        {
            7.0, 8.0, 
            9.0, 10.0, 
            11.0, 12.0
        }, true);

    deepc::Tensor<float> z = x.matmul2D(y);
    z.backward();

    std::vector<float> expected_grad_x = {15.0, 19.0, 23.0, 15.0, 19.0, 23.0};
    std::vector<float> expected_grad_y = {5.0, 5.0, 7.0, 7.0, 9.0, 9.0};

    assert(compare(x.getGrad().getFlattenedVector(), expected_grad_x));
    assert(compare(y.getGrad().getFlattenedVector(), expected_grad_y));

    std::cout << "test_grad_matmul2D passed!" << std::endl;
}

void test_set_value_with_sub_tensor() {
    deepc::Tensor<float> tensor({3, 3}, {
        1.0, 2.0, 3.0, 
        4.0, 5.0, 6.0, 
        7.0, 8.0, 9.0}, false);

    deepc::Tensor<float> sub_tensor({3}, {5.0, 5.0, 5.0}, false);

    tensor.setValue({0}, sub_tensor);

    std::vector<float> expected_values = {5.0, 5.0, 5.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    assert(compare(tensor.getFlattenedVector(), expected_values));

    std::cout << "test_set_value_with_sub_tensor passed!" << std::endl;
}

int main() {
    test_grad_addition();
    test_grad_multiplication();
    test_grad_power();
    test_grad_complex_expression();
    test_grad_broadcasting();
    test_grad_matmul2D();
    test_set_value_with_sub_tensor();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}