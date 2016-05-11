#ifndef TENSOR_TENSOR_BASE_H
#define TENSOR_TENSOR_BASE_H

#include <vector>

#include "tensor/shape.h"
#include "util/serializable.h"

namespace Tensor {

// This is a general tensor class.  As far as possible, transformations are
// done in-place eagerly.
//
// We also do not provide natural operators to have a more consistent interface.
// This results in a more verbose and less natural looking instruction set but
// encourages the use of less variables and avoid unnecessary copying.
//
// Some operations have restrictions on the way data is accessed. This is to
// make sure the class is used efficiently.
template <typename T = double>
class TensorBase : public Util::Serializable {
 public:
  virtual ~TensorBase() {}

  // Return the number of elements.
  size_t size() const { return sizeImpl(); }

  // Return the shape.
  const Shape& shape() const { return shapeImpl(); }

  // Access a const element.
  T at(const Address& address) const { return atConstImpl(address); }

  // Access a const element.
  T operator[](const Address& address) const { return atConstImpl(address); }

  // Access an element.
  T& operator[](const Address& address) { return atImpl(address); }

 private:
  virtual size_t sizeImpl() const = 0;
  virtual const Shape& shapeImpl() const = 0;
  virtual T atConstImpl(const Address& address) const = 0;
  virtual T& atImpl(const Address& address) = 0;
};

}  // Tensor

#endif  // TENSOR_TENSOR_BASE_H
