#include <numeric>

#include "tensor/shape.h"

namespace Tensor {

Shape::Shape(const Dims& dims) : dims_(dims) {
  if (!std::all_of(dims_.cbegin(), dims_.cend(),
                   [](size_t value) { return value >= 1; })) {
    throw std::invalid_argument("all dimension sizes must be >= 1");
  }
}

size_t nElements(const Shape& shape) {
  return shape.nDimensions() == 0 ? 0 : std::accumulate(
                                            shape.cbegin(), shape.cend(), 1ul,
                                            std::multiplies<size_t>());
}
}  // namespace Tensor
