#ifndef NEURAL_NET_TENSOR_ACCESSER_H_
#define NEURAL_NET_TENSOR_ACCESSER_H_

#include <numeric>
#include <vector>

#include "tensor/address_iterator.h"
#include "tensor/shape.h"

namespace Alexandria {

// This class relates the shape of the tensor to elements in a flat array.
class Accesser {
 public:
  using Strides = std::vector<size_t>;

  Accesser() {}
  explicit Accesser(const Shape* shape);

  // Calculate the flat index from the element address.
  size_t flatIndex(const Address& address) const {
    CHECK_EQ(address.size(), strides_.size()) << "sizes not equal";
    return std::inner_product(strides_.cbegin(), strides_.cend(),
                              address.cbegin(), 0ul);
  }

  // Calculate the address from the flat index.  This value is cached.
  Address address(size_t flat_index) const;

  // Forward iterator for addresses of the accessor shape.
  AddressIterator cbegin() const { return AddressIterator(*this, 0); }
  AddressIterator cend() const {
    return AddressIterator(*this, nElements(*shape_));
  }
  AddressIterator begin() const { return cbegin(); }
  AddressIterator end() const { return cend(); }

 private:
  friend class AddressIterator;

  Strides strides_;
  const Shape* shape_;
};

}  // namespace Alexandria
#endif
