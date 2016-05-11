#ifndef TENSOR_SPARSE_TENSOR_H
#define TENSOR_SPARSE_TENSOR_H

#include <unordered_map>

#include "tensor/tensor_base.h"
#include "util/util.h"

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
class SparseTensor : public TensorBase<T> {
 public:
  using Data = std::unordered_map<Address, T, AddressHash>;
  using Iterator = typename Data::iterator;
  using ConstIterator = typename Data::const_iterator;

  // Make an tensor with the same value.
  SparseTensor() {}

  // Make a sparse tensor
  explicit SparseTensor(const Shape& shape) : shape_(shape) {}

  // Iterators.
  ConstIterator begin() const { return data_.cbegin(); }
  ConstIterator end() const { return data_.cend(); }
  ConstIterator cbegin() const { return data_.cbegin(); }
  ConstIterator cend() const { return data_.cend(); }
  Iterator begin() { return data_.begin(); }
  Iterator end() { return data_.end(); }

 private:
  // Return the number of elements.
  size_t sizeImpl() const { return data_.size(); }

  // Return the shape.
  const Shape& shapeImpl() const { return shape_; }

  // Access an element.
  T atConstImpl(const Address& address) const {
    auto iter = data_.find(address);
    return iter != data_.end() ? iter->second : 0;
  }

  // Access an element.
  T& atImpl(const Address& address) { return data_[address]; }

  // Helper to remove zeros.
  void shrink() {
    for (auto iter = begin(); iter != end();) {
      if (Util::almostEqual(iter->second, 0))
        iter = data_.erase(iter);
      else
        ++iter;
    }
  }

  void serializeInImpl(Util::ArchiveIn& ar, size_t /*version*/) final {
    ar % shape_ % data_;
  }

  void serializeOutImpl(Util::ArchiveOut& ar) const final {
    ar % shape_ % data_;
  }
  size_t serializeOutVersionImpl() const final { return 0ul; }

  Shape shape_;
  Data data_;
};

}  // Tensor

#endif
