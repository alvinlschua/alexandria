#ifndef NEURAL_NET_TENSOR_TENSOR_DENSE_H_
#define NEURAL_NET_TENSOR_TENSOR_DENSE_H_

#include <vector>

#include "neural_net/tensor/accesser.h"
#include "neural_net/tensor/helpers.h"
#include "neural_net/tensor/shape.h"
#include "neural_net/tensor/tensor_base.h"
#include "util/rng.h"
#include "util/serializable.h"
#include "util/util.h"

namespace NeuralNet {

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

  // Construct a scalar.
  Dense() : shape_({1}), accesser_(&shape_), data_(1) {}

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

  // Construct an uninitialized TensorDense.
  explicit Dense(const Shape& shape)
      : shape_(shape), accesser_(&shape_), data_(nElements(shape)) {}

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

  void serializeInImpl(Util::ArchiveIn& ar, size_t /*version*/) final {
    ar % shape_ % data_;
    accesser_ = Accesser(&shape_);
  }

  void serializeOutImpl(Util::ArchiveOut& ar) const final {
    ar % shape_ % data_;
  }

  size_t serializeOutVersionImpl() const final { return 0ul; }

  std::unique_ptr<Tensor<T>::Base> cloneImpl() const {
    return std::make_unique<Tensor<T>::Dense>(*this);
  }

  Shape shape_;
  Accesser accesser_;
  Data data_;
};

// General multiplication of indices with the same index.
// Indices are how each dimension is mapped to the final result.  The result
// shape is determined by non-negative integers. Negative indices must be
// repeated in both indices are are summed over.  Repeated indices must have
// the
// same dimensions.  Indices must be unique.
//
// Examples:
//	General
//	R_jml = Sum_ik S_ijkl T_iklm
//	multiply(S, {-1, 0, -2, 2}, T, {-1, -2, 2, 1})
//
//	Outer product
//	R_ijkl = S_ij T_kl
//	multiply(S, {0, 1}, T, {2, 3})
//
//	Matrix multiply
//	R_ik = Sum_j S_ij T_jk
//	multiply(S, {0, -1}, T, {-1, 1})
//
//	Matrix multiply with transpose
//	R_ik = Sum_j S_ij T_kj
//	multiply(S, {0, -1}, T, {1, -1})
//
//	Element-wise multiply
//	R_ijk = S_ijk T_ijk
//	multiply(S, {0, 1, 2}, T, {0, 1, 2})
/*
template <typename T>
typename Tensor<T>::Dense multiply(const typename Tensor<T>::Dense& t1,
                                   const Indices& indices1,
                                   const typename Tensor<T>::Dense& t2,
                                   const Indices& indices2) {
  using namespace Util;
  using namespace std;

  Shape result_shape;
  Shape common_shape;

  tie(result_shape, common_shape) =
      multiplyShapes(t1.shape(), indices1, t2.shape(), indices2);
  auto result_addresser = Accesser(&result_shape);
  auto common_addresser = Accesser(&common_shape);

  auto result_address = Address(result_shape.nDimensions(), 0);

  auto result = typename Tensor<T>::Dense(
      result_shape, std::vector<T>(nElements(result_shape), 0));
  for (auto& element : result) {
    Address address1(t1.shape().nDimensions());
    Address address2(t2.shape().nDimensions());

    gather(indices1.cbegin(), indices1.cend(), result_address.cbegin(),
           address1.begin(),
           [](int index) { return index >= 0 ? index : Util::invalid_index; });

    gather(indices2.cbegin(), indices2.cend(), result_address.cbegin(),
           address2.begin(),
           [](int index) { return index >= 0 ? index : Util::invalid_index; });

    if (common_shape.nDimensions() > 0) {
      auto common_address = Address(common_shape.nDimensions(), 0);
      const auto size = nElements(common_shape);
      for (auto index = 0ul; index < size; ++index) {
        gather(indices1.cbegin(), indices1.cend(), common_address.cbegin(),
               address1.begin(), [](int idx) {
                 return idx < 0 ? -idx - 1 : Util::invalid_index;
               });

        gather(indices2.cbegin(), indices2.cend(), common_address.cbegin(),
               address2.begin(), [](int idx) {
                 return idx < 0 ? -idx - 1 : Util::invalid_index;
               });

        element += t1[address1] * t2[address2];
        common_address = common_addresser.increment(move(common_address));
      }
    } else {
      element = t1[address1] * t2[address2];
    }
    result_address = result_addresser.increment(move(result_address));
  }
  return result;
}
*/

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
/*
// Make a random tensor. TDistribution should be a RandomNumberDistribution
// concept.
template <typename TDistribution>
static TensorDense random(const Shape& shape, const TDistribution&
distribution);

// Make a uniform(-1, 1) random tensor.
static TensorDense random(const Shape& shape) {
  return random(shape, std::uniform_real_distribution(-1, 1));
}

*/

}  // Tensor

#endif  // NEURAL_NET_TENSOR_TENSOR_DENSE_H_
