#ifndef NEURAL_NET_TENSOR_ADDRESS_ITERATOR_H_
#define NEURAL_NET_TENSOR_ADDRESS_ITERATOR_H_

#include <iterator>

#include "tensor/shape.h"

namespace NeuralNet {

// forward declare
class Accesser;

class AddressIterator : public std::iterator<std::forward_iterator_tag, Address,
                                             ptrdiff_t, Address*, Address&> {
 public:
  AddressIterator() : index_(0) {}
  AddressIterator(const Accesser& accesser, size_t index);

  bool operator==(const AddressIterator& iterator) const {
    return index_ == iterator.index_;
  }

  bool operator!=(const AddressIterator& iter) const {
    return !(operator==(iter));
  }

  const Address& operator*() const { return address_; }
  const Address* operator->() const { return &(operator*()); }

  AddressIterator& operator++();
  AddressIterator operator++(int) {
    auto iter = AddressIterator(*this);
    ++(*this);
    return iter;
  }

 private:
  const Shape* shape_;

  Address address_;
  size_t index_;
};
}

#endif  // NEURAL_NET_TENSOR_ADDRESS_ITERATOR_H_
