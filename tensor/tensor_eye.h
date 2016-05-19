#ifndef TENSOR_TENSOR_EYE_H_
#define TENSOR_TENSOR_EYE_H_

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
class Tensor<T>::Eye : public Base {
 public:
  // Make an tensor with the same value.
  Eye() {}

  // Make a sparse tensor
  explicit Eye(const Shape& shape, T value = 1)
      : shape_(shape),
        value_(value),
        size_(std::accumulate(shape.cbegin(),
                              shape.cbegin() + shape.nDimensions() / 2, 0,
                              std::times<size_t>())) {
    if (!isEyeShape(shape_)) {
      throw std::invalid_argument("shape given is not an eye shape");
    }
  }
  Eye(const Eye&) = default;
  Eye& operator=(const Eye&) = default;

  virtual ~Eye() {}

 private:
  // Return the number of non-zero elements.
  size_t sizeImpl() const { return size_; }

  // Return the shape.
  const Shape& shapeImpl() const { return shape_; }

  // Access an element.
  T atConstImpl(const Address& address) const {
    if (address.size() != shape.nDimensions() != 0) {
      throw std::invalid_argument("eye address must have even size");
    }
    return std::equal(address.cbegin(), address.cbegin() + address.size() / 2,
                      address.cbegin() + address.size() / 2)
               ? value_
               : 0;
  }

  // Access an element.
  T& atImpl(const Address& /*address*/) {
    throw std::logic_error("eye cannot be addressed by reference");
  }

  void serializeInImpl(ArchiveIn& ar, size_t /*version*/) final { ar % shape_; }

  void serializeOutImpl(ArchiveOut& ar) const final { ar % shape_; }
  size_t serializeOutVersionImpl() const final { return 0ul; }

  std::unique_ptr<Base> cloneImpl() const {
    return std::make_unique<Eye>(*this);
  }

  virtual address_iterator cbeginImpl() const {
    return accesser_iterator(*this, 0);
  }

  virtual address_iterator cendImpl() const {
    return accesser_iterator(*this, size());
  }

  Shape shape_;
  T value_;
  size_t size_;
};

}  // Alexandria

#endif
