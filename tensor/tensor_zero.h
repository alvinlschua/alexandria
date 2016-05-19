#ifndef TENSOR_TENSOR_ZERO_H_
#define TENSOR_TENSOR_ZERO_H_

#include "tensor/helpers.h"
#include "tensor/tensor.h"
#include "tensor/tensor_base.h"

namespace Alexandria {

// This is a general tensor class.  As far as possible, transformations are
// done in-place eagerly.
//
// We also do not provide natural operators to have a more consistent interface.
// This results in a more verbose and less natural looking instruction set but
// encourages the use of less variables and avoid unnecessary copying.
//
// Some operations have restrictions on the way data is accessed. This is to
// make sure the class is used efficiently.
template <typename T>
class Tensor<T>::Zero : public Base {
 public:
  // Make an tensor with the same value.
  Zero() {}

  // Make a sparse tensor
  explicit Zero(const Shape& shape) : shape_(shape) {}
  Zero(const Zero&) = default;
  Zero& operator=(const Zero&) = default;

  virtual ~Zero() {}

 private:
  // Return the number of elements.
  size_t sizeImpl() const { return 0; }

  // Return the shape.
  const Shape& shapeImpl() const { return shape_; }

  // Access an element.
  T atConstImpl(const Address& /*address*/) const { return 0; }

  // Access an element.
  T& atImpl(const Address& /*address*/) {
    throw std::logic_error("zero cannot be addressed by reference");
  }

  void serializeInImpl(ArchiveIn& ar, size_t /*version*/) final { ar % shape_; }

  void serializeOutImpl(ArchiveOut& ar) const final { ar % shape_; }
  size_t serializeOutVersionImpl() const final { return 0ul; }

  std::unique_ptr<Base> cloneImpl() const {
    return std::make_unique<Zero>(*this);
  }

  virtual address_iterator cbeginImpl() const {
    return accesser_iterator(*this, 0);
  }

  virtual address_iterator cendImpl() const {
    return accesser_iterator(*this, size());
  }

  Shape shape_;
};

}  // Alexandria

#endif
