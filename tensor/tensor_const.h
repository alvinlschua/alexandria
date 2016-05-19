#ifndef TENSOR_TENSOR_CONST_H_
#define TENSOR_TENSOR_CONST_H_

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
class Tensor<T>::Const : public Base {
 public:
  // Make an tensor with the same value.
  Const() {}

  // Make a sparse tensor
  explicit Const(const Shape& shape, T value = 1)
      : shape_(shape),
        value_(value),
        size_(std::accumulate(shape.cbegin(), shape.cend(), 1ul,
                              std::multiplies<size_t>())) {
    CHECK_NE(value, 0) << "value should not be zero";
  }
  Const(const Const&) = default;
  Const& operator=(const Const&) = default;

  virtual ~Const() {}

 private:
  size_t sizeImpl() const final { return size_; }

  const Shape& shapeImpl() const final { return shape_; }

  T atImpl(const Address& /*address*/) const final {
    return value_;
  }

  void setImpl(const Address&, T, std::function<T(T, T)>) final {
    throw std::invalid_argument("cannot set a const tensor");
  }

  AddressIterator beginImpl() const {
    return AddressIterator(0ul, Address(this->shape().nDimensions(), 0),
                           &value_, [this](size_t /*index*/, Address& address) {
                             address =
                                 increment(std::move(address), this->shape());
                             return &value_;
                           });
  }

  AddressIterator endImpl() const { return AddressIterator(this->size()); }

  void serializeInImpl(ArchiveIn& ar, size_t /*version*/) final { ar % shape_; }

  void serializeOutImpl(ArchiveOut& ar) const final { ar % shape_; }
  size_t serializeOutVersionImpl() const final { return 0ul; }

  std::unique_ptr<Base> cloneImpl() const {
    return std::make_unique<Const>(*this);
  }

  Shape shape_;
  T value_;
  size_t size_;
};

}  // Alexandria

#endif
