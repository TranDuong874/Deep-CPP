#ifndef LAYER_H
#define LAYEE_H

#include <vector>
#include <memory>
#include "tensor.h"

namespace deepc {
    class Layer {
        public:
            virtual Tensor<float> forward(Tensor<float>& input) = 0;
            virtual void backward() = 0;
            virtual void getInfo() = 0;
    };

    using LayerPtr = std::shared_ptr<Layer>;
}
#endif