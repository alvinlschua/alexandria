#ifndef TENSOR_HELPERS_H
#define TENSOR_HELPERS_H

#include <tuple>
#include "tensor/shape.h"

namespace Tensor {

// Are indices unique?
bool indicesUnique(Indices indices);

// Returns the result and common shapes needed for tensor multiplication in that
// order.
//
// The result shape refers the result of the multiplication.
// The common shape refers the shape associated with common indices to sum over.
std::pair<Shape, Shape> multiplyShape(const Shape& shape1,
                                      const Indices& indices1,
                                      const Shape& shape2,
                                      const Indices& indices2);

// Returns the separated result and common of both indices needed for tensor
// multiplication in the form {result_indices1, common_indices1,
// result_indices2, common_indices2}.
//
// The multiplication algorithm requires that both multiplicand tensor address
// are constructed from the address of the result tensor and the address of
// indices the are common to both shapes.
std::tuple<Indices, Indices, Indices, Indices> multiplyIndices(
    const Indices& indices1, const Indices& indices2);

std::pair<Shape, Shape> multiplyShapes(const Shape& shape1,
                                       const Indices& indices1,
                                       const Shape& shape2,
                                       const Indices& indices2);
}  // Tensor

#endif
