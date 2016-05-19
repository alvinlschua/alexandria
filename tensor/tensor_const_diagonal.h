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
class Tensor<T>::ConstDiagonal : public Base {
 public:
  // Make an tensor with the same value.
  ConstDiagonal() {}

  // Make a sparse tensor
  explicit ConstDiagonal(const Shape& shape, T value = 1)
      : shape_(shape),
        value_(value),
        size_(std::accumulate(shape.cbegin(),
                              shape.cbegin() + shape.nDimensions() / 2, 1ul,
                              std::multiplies<size_t>())) {
    if (!isEyeShape(shape_)) {
      throw std::invalid_argument("shape given is not an eye shape");
    }

    std::vector<size_t> dims(shape_.cbegin(),
                             shape_.cbegin() + shape_.nDimensions() / 2);
    half_shape_ = Shape(dims);
  }
  ConstDiagonal(const ConstDiagonal&) = default;
  ConstDiagonal& operator=(const ConstDiagonal&) = default;

  virtual ~ConstDiagonal() {}

 private:
  size_t sizeImpl() const final { return size_; }

  const Shape& shapeImpl() const final { return shape_; }

  T atImpl(const Address& address) const final {
    if (address.size() != this->shape().nDimensions()) {
      throw std::invalid_argument("eye address must have even size");
    }
    return std::equal(address.cbegin(), address.cbegin() + address.size() / 2,
                      address.cbegin() + address.size() / 2)
               ? value_
               : 0;
  }

  void setImpl(const Address&, T, std::function<T(T, T)>) final {
    throw std::invalid_argument("cannot set eye");
  }

  AddressIterator beginImpl() const {
    return AddressIterator(
        0ul, Address(this->shape().nDimensions(), 0), &value_,
        [this](size_t /*index*/, Address& address) {
          auto half_address =
              Address(address.cbegin(), address.cbegin() + address.size() / 2);
          half_address = increment(std::move(half_address), half_shape_);
          auto iter = std::copy(half_address.cbegin(), half_address.cend(),
                                address.begin());
          std::copy(half_address.cbegin(), half_address.cend(), iter);
          return &value_;
        });
  }

  AddressIterator endImpl() const { return AddressIterator(this->size()); }

  void serializeInImpl(ArchiveIn& ar, size_t /*version*/) final { ar % shape_; }

  void serializeOutImpl(ArchiveOut& ar) const final { ar % shape_; }
  size_t serializeOutVersionImpl() const final { return 0ul; }

  std::unique_ptr<Base> cloneImpl() const {
    return std::make_unique<ConstDiagonal>(*this);
  }

  Shape shape_;
  Shape half_shape_;
  T value_;
  size_t size_;
};

}  // Alexandria

#endif
