#ifndef TENSOR_HELPERS_H
#define TENSOR_HELPERS_H

#include <tuple>
#include "neural_net/tensor/shape.h"

namespace NeuralNet {

// Are indices unique?
bool indicesUnique(Indices indices);

// Returns the result and common shapes needed for tensor multiplication in that
// order.
//
// The result shape refers the result of the multiplication.
// The common shape refers the shape associated with common indices to sum over.
std::pair<Shape, Shape> multiplyShapes(const Shape& shape1,
                                       const Indices& indices1,
                                       const Shape& shape2,
                                       const Indices& indices2);

}  // Tensor

#endif
