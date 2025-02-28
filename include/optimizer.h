#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include "layer.h"

namespace deepc {
    class Optimizer {
        protected:
            float lr;  
        
        public:
            Optimizer(float learning_rate) : lr(learning_rate) {}
        
            virtual void step(std::vector<Layer*>* &layers) = 0; 
        };
}


#endif