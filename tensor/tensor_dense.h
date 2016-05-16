#ifndef NEURAL_NET_TENSOR_TENSOR_DENSE_H_
#define NEURAL_NET_TENSOR_TENSOR_DENSE_H_

#include <vector>

#include "tensor/accesser.h"
#include "tensor/helpers.h"
#include "tensor/shape.h"
#include "tensor/tensor_base.h"
#include "util/rng.h"
#include "util/serializable.h"
#include "util/util.h"

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
class Tensor<T>::Dense : public Base {
 public:
  using Data = std::vector<T>;
  using Iterator = typename Data::iterator;
  using ConstIterator = typename Data::const_iterator;

  Dense() {}

  // Construct an uninitialized Tensor<T>::Dense.
  explicit Dense(const Shape& shape)
      : shape_(shape), accesser_(&shape_), data_(nElements(shape)) {}

  Dense(const Shape& shape, Data data)
      : shape_(shape), accesser_(&shape_), data_(std::move(data)) {}

  Dense(const Dense& tensor)
      : shape_(tensor.shape_), accesser_(&shape_), data_(tensor.data_) {}

  Dense& operator=(const Dense& tensor) {
    Dense tensor1(tensor);
    std::swap(*this, tensor1);
    return *this;
  }

  Dense(Dense&& tensor)
      : shape_(std::move(tensor.shape_)),
        accesser_(&shape_),
        data_(std::move(tensor.data_)) {}

  Dense& operator=(Dense&& tensor) {
    std::swap(shape_, tensor.shape_);
    std::swap(data_, tensor.data_);
    accesser_ = Accesser(&shape_);
    return *this;
  }

  // Iterators.
  ConstIterator begin() const { return data_.cbegin(); }
  ConstIterator end() const { return data_.cend(); }
  ConstIterator cbegin() const { return data_.cbegin(); }
  ConstIterator cend() const { return data_.cend(); }
  Iterator begin() { return data_.begin(); }
  Iterator end() { return data_.end(); }

 private:
  // Return the number of elements.
  size_t sizeImpl() const final { return data_.size(); }

  // Return the shape.
  const Shape& shapeImpl() const final { return shape_; }

  // Access an element.
  T atConstImpl(const Address& address) const {
    return data_.at(accesser_.flatIndex(address));
  }

  // Access an element.
  T& atImpl(const Address& address) {
    return data_[accesser_.flatIndex(address)];
  }

  void serializeInImpl(ArchiveIn& ar, size_t /*version*/) final {
    ar % shape_ % data_;
    accesser_ = Accesser(&shape_);
  }

  void serializeOutImpl(ArchiveOut& ar) const final { ar % shape_ % data_; }

  size_t serializeOutVersionImpl() const final { return 0ul; }

  std::unique_ptr<Tensor<T>::Base> cloneImpl() const {
    return std::make_unique<Tensor<T>::Dense>(*this);
  }

  Shape shape_;
  Accesser accesser_;
  Data data_;
};

template <typename T>
std::ostream& operator<<(std::ostream& out,
                         const typename Tensor<T>::Dense& t) {
  out << t.shape() << "{";
  if (t.size() > 12) {
    out << " " << t.size() << " elements ";
  } else {
    auto iter = t.cbegin();
    for (; iter != t.cend() - 1; ++iter) {
      out << *iter << ", ";
    }
    out << *iter;
  }
  out << "}";

  return out;
}

}  // Tensor

#endif  // NEURAL_NET_TENSOR_TENSOR_DENSE_H_
