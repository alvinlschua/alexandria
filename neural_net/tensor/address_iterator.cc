#include <iterator>

#include "neural_net/tensor/accesser.h"
#include "neural_net/tensor/address_iterator.h"

namespace NeuralNet {

AddressIterator::AddressIterator(const Accesser& accessor, size_t index)
    : shape_(accessor.shape_),
      address_(accessor.address(index)),
      index_(index) {}

AddressIterator& AddressIterator::operator++() {
  ++index_;
  ++address_.back();

  auto carry = 0ul;
  auto iter = address_.rbegin();
  auto dim_iter = shape_->crbegin();

  for (; iter != address_.rend(); ++iter, ++dim_iter) {
    auto value = *iter;
    const auto dim = *dim_iter;
    value += carry;
    carry = value / dim;
    *iter = value % dim;
    if (carry == 0) break;
  }
  return *this;
}
}  // namespace NeuralNet
