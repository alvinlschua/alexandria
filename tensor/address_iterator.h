#ifndef NEURAL_NET_TENSOR_ADDRESS_ITERATOR_H_
#define NEURAL_NET_TENSOR_ADDRESS_ITERATOR_H_

#include <iterator>

namespace Alexandria {

template <typename T>
class Tensor<T>::AddressIterator
    : public std::iterator<std::forward_iterator_tag,
                           std::pair<const Address, const T&>, ptrdiff_t,
                           std::pair<const Address, const T&>*,
                           std::pair<const Address, const T&>&> {
 public:
  using ValueType = typename AddressIterator::value_type;

  explicit AddressIterator(size_t index = 0) : index_(index), value_(nullptr) {}

  AddressIterator(size_t index, const Address& address, const T* value,
                  std::function<const T*(size_t, Address&)> increment)
      : index_(index),
        address_(address),
        value_(value),
        increment_(increment) {}

  bool operator==(const AddressIterator& iterator) const {
    return index_ == iterator.index_;
  }

  bool operator!=(const AddressIterator& iter) const {
    return !(operator==(iter));
  }

  ValueType operator*() const { return ValueType(address_, *value_); }
  const Address* operator->() const { return &(operator*()); }

  AddressIterator& operator++() {
    ++index_;
    value_ = increment_(index_, address_);
    return *this;
  }
  AddressIterator operator++(int) {
    auto iter = AddressIterator(*this);
    ++(*this);
    return iter;
  }

 private:
  size_t index_;
  Address address_;
  const T* value_;
  std::function<const T*(size_t, Address&)> increment_;
};
}

#endif  // NEURAL_NET_TENSOR_ADDRESS_ITERATOR_H_
