#ifndef TENSOR_HELPERS_H
#define TENSOR_HELPERS_H

#include "tensor/shape.h"
#include <tuple>

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

// Returns the combined shape.
Shape combineShapes(const Shape& shape1, const Shape& shape2);

inline bool isEyeShape(const Shape& shape) {
  if (shape.nDimensions() % 2 != 0) {
    return false;
  }

  return std::equal(shape.cbegin(), shape.cbegin() + shape.nDimensions() / 2,
                    shape.cbegin() + shape.nDimensions() / 2);
}

}  // Tensor

#endif
