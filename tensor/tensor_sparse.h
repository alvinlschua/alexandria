#ifndef TENSOR_SPARSE_TENSOR_H
#define TENSOR_SPARSE_TENSOR_H

#include <map>

#include "tensor/accesser.h"
#include "tensor/address_iterator.h"
#include "tensor/helpers.h"
#include "tensor/tensor_base.h"
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
class Tensor<T>::Sparse : public Base {
 public:
  using Data = std::unordered_map<Address, T, AddressHash>;
  using Iterator = typename Data::const_iterator;

  // Make an tensor with the same value.
  Sparse() {}

  // Make a sparse tensor
  explicit Sparse(const Shape& shape) : shape_(shape) {}

  explicit Sparse(const Shape& shape, Data data)
      : shape_(shape), data_(std::move(data)) {}

  Sparse(const Sparse&) = default;
  Sparse& operator=(const Sparse&) = default;

  virtual ~Sparse() {}

  // Data.
  const Data& data() const { return data_; }
  Data& data() { return data_; }

 private:
  size_t sizeImpl() const { return data_.size(); }

  const Shape& shapeImpl() const { return shape_; }

  T atImpl(const Address& address) const {
    auto iter = data_.find(address);
    return iter != data_.end() ? iter->second : 0;
  }

  void setImpl(const Address& address, T value,
               std::function<T(T, T)> fn) final {
    auto iter = data_.find(address);
    auto result = fn(iter == data_.end() ? 0 : iter->second, value);
    if (almostEqual(result, 0)) {
      if (iter != data_.end()) data_.erase(address);
      return;
    }

    if (iter != data_.cend()) {
      iter->second = result;
    } else {
      data_[address] = result;
    }
  }

  AddressIterator beginImpl() const final {
    return AddressIterator(
        0ul, data_.size() == 0 ? Address() : data_.cbegin()->first,
        data_.size() == 0 ? nullptr : &(data_.cbegin()->second),
        [this](size_t index, Address& address) {
          auto iter = data_.cbegin();
          for (auto idx = 0ul; idx < index; ++idx) ++iter;
          address = iter == data_.cend() ? Address() : iter->first;
          return iter == data_.cend() ? nullptr : &(iter->second);
        });
  }

  AddressIterator endImpl() const final {
    return AddressIterator(this->size());
  }

  void serializeInImpl(ArchiveIn& ar, size_t /*version*/) final {
    ar % shape_ % data_;
  }

  void serializeOutImpl(ArchiveOut& ar) const final { ar % shape_ % data_; }
  size_t serializeOutVersionImpl() const final { return 0ul; }

  std::unique_ptr<Base> cloneImpl() const {
    return std::make_unique<Sparse>(*this);
  }

  Shape shape_;
  Data data_;
};

}  // Alexandria

#endif
