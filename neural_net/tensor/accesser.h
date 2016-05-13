#ifndef NEURAL_NET_TENSOR_ACCESSER_H_
#define NEURAL_NET_TENSOR_ACCESSER_H_

#include <numeric>
#include <vector>

#include "neural_net/tensor/shape.h"
#include "util/serializable.h"

namespace NeuralNet {

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

  // Increments the given address.  Note that address can be moved in.
  //
  // This calculation should be faster than address as it terminates early once
  // carry = 0.
  Address increment(Address address, size_t amount = 1ul);

 private:
  Strides strides_;
  const Shape* shape_;
};

}  // namespace NeuralNet
#endif
