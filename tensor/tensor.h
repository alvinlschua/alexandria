#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include <vector>

#include "tensor/tensor_base.h"
#include "tensor/addresser.h"
#include "tensor/shape.h"
#include "tensor/helpers.h"
#include "util/rng.h"
#include "util/serializable.h"
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
class Tensor : public TensorBase<T> {
 public:
  using Data = std::vector<T>;
  using Iterator = typename Data::iterator;
  using ConstIterator = typename Data::const_iterator;

  // Make an tensor with the same value.
  static Tensor fill(const Shape& shape, T value) {
    Data data(nElements(shape), value);
    return Tensor(shape, std::move(data));
  }

  // Make a tensor of zeros.
  static Tensor zeros(const Shape& shape) { return fill(shape, 0); }

  // Make a tensor of ones.
  static Tensor ones(const Shape& shape) { return fill(shape, 1); }

  // Make a random tensor. TDistribution should be a RandomNumberDistribution
  // concept.
  /*
  template <typename TDistribution>
  static Tensor random(const Shape& shape, const TDistribution& distribution);

  // Make a uniform(-1, 1) random tensor.
  static Tensor random(const Shape& shape) {
    return random(shape, std::uniform_real_distribution(-1, 1));
  }
  */

  // Make a tensor using a generator.
  static Tensor generate(const Shape& shape,
                         std::function<T(Address)> generator) {
    Data data(nElements(shape));
    Addresser addresser(&shape);
    for (auto index = 0ul; index < data.size(); ++index) {
      data[index] = generator(addresser.address(index));
    }
    return Tensor(shape, std::move(data));
  }

  // Default constructor
  Tensor() {}

  Tensor(const Tensor& tensor)
      : shape_(tensor.shape_), addresser_(&shape_), data_(tensor.data_) {}

  Tensor& operator=(const Tensor& tensor) {
    Tensor tensor1(tensor);
    std::swap(*this, tensor1);
    return *this;
  }

  Tensor(Tensor&& tensor)
      : shape_(std::move(tensor.shape_)),
        addresser_(&shape_),
        data_(std::move(tensor.data_)) {}

  Tensor& operator=(Tensor&& tensor) {
    std::swap(shape_, tensor.shape_);
    std::swap(data_, tensor.data_);
    addresser_ = Addresser(&shape_);
    return *this;
  }

  // Construct an uninitialized Tensor.
  explicit Tensor(const Shape& shape)
      : shape_(shape), addresser_(&shape_), data_(nElements(shape)) {}

  explicit Tensor(std::vector<T> data)
      : shape_({data.size()}), addresser_(&shape_), data_(std::move(data)) {}

  explicit Tensor(std::vector<std::vector<T>> values)
      : shape_({values.size(), values.front().size()}),
        addresser_(&shape_),
        data_(nElements(shape_)) {
    auto index = 0ul;
    for (auto iter1 = values.begin(); iter1 != values.end(); ++iter1) {
      for (auto iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2) {
        data_[index++] = *iter2;
      }
    }
  }

  explicit Tensor(std::vector<std::vector<std::vector<T>>> values)
      : shape_({values.size(), values.front().size(),
                values.front().front().size()}),
        addresser_(&shape_),
        data_(nElements(shape_)) {
    auto index = 0ul;
    for (auto iter1 = values.begin(); iter1 != values.end(); ++iter1) {
      for (auto iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2) {
        for (auto iter3 = iter2->begin(); iter3 != iter2->end(); ++iter3) {
          data_[index++] = *iter3;
        }
      }
    }
  }

  // Iterators.
  ConstIterator begin() const { return data_.cbegin(); }
  ConstIterator end() const { return data_.cend(); }
  ConstIterator cbegin() const { return data_.cbegin(); }
  ConstIterator cend() const { return data_.cend(); }
  Iterator begin() { return data_.begin(); }
  Iterator end() { return data_.end(); }

 private:
  Tensor(const Shape& shape, Data data)
      : shape_(shape), addresser_(&shape_), data_(std::move(data)) {}

  // Return the number of elements.
  size_t sizeImpl() const final { return data_.size(); }

  // Return the shape.
  const Shape& shapeImpl() const final { return shape_; }

  // Access an element.
  T atConstImpl(const Address& address) const {
    return data_.at(addresser_.flatIndex(address));
  }

  // Access an element.
  T& atImpl(const Address& address) {
    return data_[addresser_.flatIndex(address)];
  }

  void serializeInImpl(Util::ArchiveIn& ar, size_t /*version*/) final {
    ar % shape_ % data_;
    addresser_ = Addresser(&shape_);
  }

  void serializeOutImpl(Util::ArchiveOut& ar) const final {
    ar % shape_ % data_;
  }

  size_t serializeOutVersionImpl() const final { return 0ul; }


  Shape shape_;
  Addresser addresser_;
  Data data_;
};

template <typename T>
bool operator==(const Tensor<T>& t1, const Tensor<T>& t2) {
  return t1.shape() == t2.shape() &&
         std::equal(t1.cbegin(), t1.cend(), t2.cbegin());
}

template <typename T>
bool operator!=(const Tensor<T>& t1, const Tensor<T>& t2) {
  return !(t1 == t2);
}

template <typename T>
Tensor<T> apply(Tensor<T> result, const Tensor<T>& t2,
                std::function<T(T, T)> fn) {
  if (result.shape() != t2.shape()) {
    throw std::invalid_argument("shapes are not the same");
  }
  std::transform(result.cbegin(), result.cend(), t2.cbegin(), result.begin(),
                 fn);

  return result;
}

template <typename T>
Tensor<T> apply(Tensor<T> result, std::function<T(T)> fn) {
  std::transform(result.cbegin(), result.cend(), result.cbegin(), fn);
  return result;
}

template <typename T>
inline Tensor<T> operator+(const Tensor<T>& t1, const Tensor<T>& t2) {
  return plus(t1, t2);
}

template <typename T>
inline Tensor<T> plus(Tensor<T> result, const Tensor<T>& t2) {
  return apply<T>(std::move(result), t2, [](T x, T y) { return x + y; });
}

template <typename T>
inline Tensor<T> operator-(const Tensor<T>& t1, const Tensor<T>& t2) {
  return minus(t1, t2);
}

template <typename T>
inline Tensor<T> minus(Tensor<T> result, const Tensor<T>& t2) {
  return apply<T>(std::move(result), t2, [](T x, T y) { return x - y; });
}

template <typename T>
inline Tensor<T> multiply(Tensor<T> result, T value) {
  return apply(std::move(result), [value](T x) { return value * x; });
}

template <typename T>
inline Tensor<T> operator*(const Tensor<T>& t1, T value) {
  return multiply(t1, value);
}

template <typename T>
inline Tensor<T> operator*(T value, const Tensor<T>& t1) {
  return multiply(t1, value);
}

template <typename T>
inline Tensor<T> divide(Tensor<T> result, T value) {
  return apply(std::move(result), [value](T x) { return x / value; });
}

template <typename T>
inline Tensor<T> operator/(const Tensor<T>& t1, T value) {
  return divide(t1, value);
}

template <typename T>
inline Tensor<T> unaryMinus(Tensor<T> result) {
  return apply(std::move(result), [](T x) { return -x; });
}

template <typename T>
inline Tensor<T> operator-(const Tensor<T>& result) {
  return unaryMinus(result);
}

// General multiplication of indices with the same index.
// Indices are how each dimension is mapped to the final result.  The result
// shape is determined by non-negative integers. Negative indices must be
// repeated in both indices are are summed over.  Repeated indices must have the
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
template <typename T>
Tensor<T> multiply(const Tensor<T>& tensor1, const Indices& indices1,
                   const Tensor<T>& tensor2, const Indices& indices2) {
  using namespace Util;
  using namespace std;

  Shape result_shape;
  Shape common_shape;

  tie(result_shape, common_shape) =
      multiplyShapes(tensor1.shape(), indices1, tensor2.shape(), indices2);
  auto result_addresser = Addresser(&result_shape);
  auto common_addresser = Addresser(&common_shape);

  auto result_address = Address(result_shape.nDimensions(), 0);

  auto result = Tensor<>::zeros(result_shape);
  for (auto& element : result) {
    Address address1(tensor1.shape().nDimensions());
    Address address2(tensor2.shape().nDimensions());

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

        element += tensor1[address1] * tensor2[address2];
        common_address = common_addresser.increment(move(common_address));
      }
    } else {
      element = tensor1[address1] * tensor2[address2];
    }
    result_address = result_addresser.increment(move(result_address));
  }
  return result;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& t) {
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

#endif  // TENSOR_TENSOR_H
