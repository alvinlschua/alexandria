#include <algorithm>
#include <numeric>

#include "tensor/addresser.h"

namespace Tensor {

Addresser::Addresser(const Shape* shape) : shape_(shape) {
  using namespace std;

  if (shape_->nDimensions() == 0ul) return;

  strides_.resize(shape_->nDimensions());
  partial_sum(shape_->crbegin(), shape_->crend() - 1, strides_.rbegin() + 1,
              [](size_t tot, size_t val) { return tot * val; });
  strides_.back() = 1ul;
}

Address Addresser::address(size_t flat_index) const {
  Address result(strides_.size());
  for (auto index = 0ul; index < strides_.size(); ++index) {
    result[index] =
        (*shape_)[index] == 1ul ? 0ul : flat_index / strides_.at(index);
    flat_index = flat_index % strides_.at(index);
  }
  return result;
}

Address Addresser::increment(Address address, size_t amount) {
  CHECK(amount >= 1) << "increment amount should be >= 1";
  CHECK(address.size() == shape_->nDimensions())
      << "address and shape sizes inconsistent";

  address.back() += amount;

  auto carry = 0ul;
  auto iter = address.rbegin();
  auto dim_iter = shape_->crbegin();

  for (; iter != address.rend(); ++iter, ++dim_iter) {
    auto value = *iter;
    const auto dim = *dim_iter;
    value += carry;
    carry = value / dim;
    *iter = value % dim;
    if (carry == 0) break;
  }

  return address;
}
}  // namespace Tensor
