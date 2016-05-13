#ifndef NEURAL_NET_TENSOR_H
#define NEURAL_NET_TENSOR_H

#include <vector>

#include "neural_net/tensor/accesser.h"
#include "neural_net/tensor/shape.h"
#include "neural_net/tensor/helpers.h"
#include "util/clonable.h"
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
class Tensor : public Util::Serializable {
 public:
  class Base;
  class Dense;
  class Sparse;
  class Eye;
  class Zeros;

  using Ptr = std::unique_ptr<Base>;

  virtual ~Tensor() {}

  Tensor() {}

  Tensor(const Tensor& tensor) : ptr_(tensor.ptr_->clone()) {}
  Tensor& operator=(const Tensor& tensor) {
    ptr_ = std::move(tensor.ptr_->clone());
    return *this;
  }

  explicit Tensor(const Dense& tensor) : ptr_(tensor.clone()) {}
  explicit Tensor(const Sparse& tensor) : ptr_(tensor.clone()) {}

  explicit Tensor(const std::vector<T>& data)
      : ptr_(Dense(Shape({data.size()}), data).clone()) {}
  explicit Tensor(const std::vector<std::vector<T>>& data);
  explicit Tensor(const std::vector<std::vector<std::vector<T>>>& data);

  static Tensor<T> fill(const Shape& shape, T value);
  static Tensor zeros(const Shape& shape);
  static Tensor ones(const Shape& shape);

  // uninitialized dense
  static Tensor dense(const Shape& shape);
  static Tensor sparse(const Shape& shape);
  static Tensor generate(const Shape& shape, std::function<T(Address)> fn);

  // shape
  static Tensor outerProductEye(const Shape& half_shape);
  static Tensor outerProductZeros(const Shape& half_shape);

  // Return the number of elements.
  size_t size() const { return ptr_->size(); }

  // Return the shape.
  const Shape& shape() const { return ptr_->shape(); }

  // Access a const element.
  T at(const Address& address) const { return ptr_->at(address); }

  // Access a const element.
  T operator[](const Address& address) const { return ptr_->at(address); }

  // Access an element.
  T& operator[](const Address& address) { return (*ptr_)[address]; }

  // Is this of type U?
  template <typename U>
  bool isType() const {
    return pointer<U>() != nullptr;
  }

  // Cast this to type U.  A nullptr is return is it cannot be cast.
  template <typename U>
  U* pointer() const {
    return dynamic_cast<U*>(ptr_.get());
  }

  // Cast this to reference of type T.  This fails if isType<T> is false.
  template <typename U>
  U& reference() const {
    auto result = pointer<U>();
    if (result == nullptr) {
      throw std::logic_error(std::string("unable to cast to ") +
                             typeid(T).name() + "in castReference");
    }
    return *result;
  }

 private:
  using ArchiveIn = Util::ArchiveIn;
  using ArchiveOut = Util::ArchiveOut;

  Tensor(Ptr ptr) : ptr_(std::move(ptr)) {}

  void serializeInImpl(ArchiveIn& ar, size_t /*version*/) final;
  void serializeOutImpl(ArchiveOut& ar) const final;
  size_t serializeOutVersionImpl() const final { return 0ul; }

  Ptr ptr_;
};

template <typename T>
bool operator==(const Tensor<T>& t1, const Tensor<T>& t2) {
  if (t1.template isType<typename Tensor<T>::Dense>() &&
      t2.template isType<typename Tensor<T>::Dense>()) {
    auto x = t1.template reference<typename Tensor<T>::Dense>();
    auto y = t2.template reference<typename Tensor<T>::Dense>();
    return x.shape() == y.shape() &&
           std::equal(x.cbegin(), x.cend(), y.cbegin());
  } else {
    throw Util::unimplemented_exception("unimplemented");
  }
}

template <typename T>
bool operator!=(const Tensor<T>& t1, const Tensor<T>& t2) {
  return !(t1 == t2);
}

template <typename T>
Tensor<T> Tensor<T>::generate(const Shape& shape,
                              std::function<T(Address)> fn) {
  Dense result(shape);
  const auto size = nElements(shape);
  Address address(shape.nDimensions(), 0);
  Accesser accesser(&shape);
  for (auto index = 0ul; index < size;
       ++index, address = accesser.increment(std::move(address))) {
    result[address] = fn(address);
  }
  return Tensor(result);
}

template <typename T>
Tensor<T> Tensor<T>::sparse(const Shape& shape) {
  return Tensor(Sparse(shape));
}

template <typename T>
Tensor<T> Tensor<T>::dense(const Shape& shape) {
  return Tensor(Dense(shape));
}

template <typename T>
Tensor<T>::Tensor(const std::vector<std::vector<T>>& data) {
  Shape shape({data.size(), data.front().size()});
  std::vector<T> result(nElements(shape));

  auto index = 0ul;
  for (auto iter1 = data.begin(); iter1 != data.end(); ++iter1) {
    for (auto iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2) {
      CHECK_EQ(iter1->size(), shape[1]) << "sizes inconsistent";
      result[index++] = *iter2;
    }
  }
  ptr_ = Dense(shape, result).clone();
}

template <typename T>
Tensor<T>::Tensor(const std::vector<std::vector<std::vector<T>>>& data) {
  Shape shape({data.size(), data.front().size(), data.front().front().size()});
  std::vector<T> result(nElements(shape));

  auto index = 0ul;
  for (auto iter1 = data.begin(); iter1 != data.end(); ++iter1) {
    for (auto iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2) {
      CHECK_EQ(iter1->size(), shape[1]) << "sizes inconsistent";
      for (auto iter3 = iter2->begin(); iter3 != iter2->end(); ++iter3) {
        CHECK_EQ(iter2->size(), shape[2]) << "sizes inconsistent";
        result[index++] = *iter3;
      }
    }
  }
  ptr_ = Dense(shape, result).clone();
}

Shape outerProductShape(const Shape& shape);
Shape outerProductShape(const Shape& shape) {
  Indices index1(shape.nDimensions());
  Indices index2(shape.nDimensions());

  std::iota(index1.begin(), index1.end(), 0);
  std::iota(index2.begin(), index2.end(), shape.nDimensions());

  Shape result;
  std::tie(result, std::ignore) = multiplyShapes(shape, index1, shape, index2);
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::outerProductZeros(const Shape& shape) {
  return Tensor<T>::sparse(outerProductShape(shape));
}

// Makes an eye with shape x shape, so that e_ij,lm = delta_il delta_jm
template <typename T>
Tensor<T> Tensor<T>::outerProductEye(const Shape& shape) {
  Shape resultShape = outerProductShape(shape);

  Sparse result(resultShape);
  Accesser accesser(&shape);
  Address address(shape.nDimensions(), 0);

  auto size = nElements(shape);
  for (auto index = 0ul; index < size;
       ++index, address = accesser.increment(std::move(address))) {
    auto resultAddress = address;
    resultAddress.resize(2 * shape.nDimensions());

    std::copy(resultAddress.cbegin(), resultAddress.cend(),
              resultAddress.begin() + static_cast<int>(shape.nDimensions()));

    result[resultAddress] = 1;
  }

  return Tensor(result);
}

template <typename T>
void Tensor<T>::serializeInImpl(Util::ArchiveIn& ar, size_t /*version*/) {
  bool isDense = false;
  ar % isDense;
  ptr_ = std::make_unique<Dense>();
  ar % (*ptr_);
}

template <typename T>
void Tensor<T>::serializeOutImpl(Util::ArchiveOut& ar) const {
  bool isDense = this->isType<Dense>();
  ar % isDense % (*ptr_);
}

template <typename T>
Tensor<T> Tensor<T>::fill(const Shape& shape, T value) {
  auto size = nElements(shape);
  auto data = std::vector<T>(size, value);

  return Tensor<T>(Dense(shape, std::move(data)).clone());
}

template <typename T>
Tensor<T> Tensor<T>::ones(const Shape& shape) {
  return fill(shape, 1);
}

template <typename T>
Tensor<T> Tensor<T>::zeros(const Shape& shape) {
  return fill(shape, 0);
}

// Note: Moving apply functionality is worth it only if it is treated as a
// member function.
template <typename T>
Tensor<T> apply(Tensor<T> t, std::function<T(T)> fn) {
  if (t.template isType<typename Tensor<T>::Dense>()) {
    auto& temp = t.template reference<typename Tensor<T>::Dense>();
    std::transform(temp.cbegin(), temp.cend(), temp.begin(), fn);
  } else if (t.template isType<typename Tensor<T>::Sparse>()) {
    auto& temp = t.template reference<typename Tensor<T>::Sparse>();
    for (auto& item : temp) {
      item.second = fn(item.second);
    }
    temp.defaultValue(fn(temp.defaultValue()));
  } else {
    throw Util::unimplemented_exception(
        "apply binary not implemented for type");
  }

  return t;
}

// Note: Moving apply functionality into derived classes is worth it only if
// it
// is treated as a member function.
template <typename T>
Tensor<T> apply(Tensor<T> t1, const Tensor<T>& t2, std::function<T(T, T)> fn) {
  if (t1.shape() != t2.shape()) {
    throw std::invalid_argument("shapes are not the same");
  }

  if (t1.template isType<typename Tensor<T>::Dense>() &&
      t2.template isType<typename Tensor<T>::Dense>()) {
    auto& temp1 = t1.template reference<typename Tensor<T>::Dense>();
    auto& temp2 = t2.template reference<typename Tensor<T>::Dense>();
    std::transform(temp1.cbegin(), temp1.cend(), temp2.cbegin(), temp1.begin(),
                   fn);
  } else {
    throw Util::unimplemented_exception(
        "apply binary not implemented for type");
  }

  return t1;
}

template <typename T>
inline Tensor<T> unaryMinus(Tensor<T> t) {
  return apply<T>(std::move(t), [](T x) { return -x; });
}

template <typename T>
inline Tensor<T> operator-(const Tensor<T>& t) {
  return unaryMinus(t);
}

template <typename T>
inline Tensor<T> multiply(Tensor<T> t, T value) {
  return apply<T>(std::move(t), [value](T x) { return value * x; });
}

template <typename T>
inline Tensor<T> operator*(const Tensor<T>& t, T value) {
  return multiply(t, value);
}

template <typename T>
inline Tensor<T> operator*(T value, const Tensor<T>& t) {
  return multiply(t, value);
}

template <typename T>
inline Tensor<T> divide(Tensor<T> t, T value) {
  return apply<T>(std::move(t), [value](T x) { return x / value; });
}

template <typename T>
inline Tensor<T> operator/(const Tensor<T>& t, T value) {
  return divide(t, value);
}

template <typename T>
inline Tensor<T> plus(Tensor<T> t1, const Tensor<T>& t2) {
  return apply<T>(std::move(t1), t2, [](T x, T y) { return x + y; });
}

template <typename T>
inline Tensor<T> operator+(const Tensor<T>& t1, const Tensor<T>& t2) {
  return plus(t1, t2);
}

template <typename T>
inline Tensor<T> minus(Tensor<T> t1, const Tensor<T>& t2) {
  return apply<T>(std::move(t1), t2, [](T x, T y) { return x - y; });
}

template <typename T>
inline Tensor<T> operator-(const Tensor<T>& t1, const Tensor<T>& t2) {
  return minus(t1, t2);
}

template <typename T>
typename Tensor<T>::Dense multiply(const typename Tensor<T>::Dense& t1,
                                   const Indices& indices1,
                                   const typename Tensor<T>::Dense& t2,
                                   const Indices& indices2);

/*
template <typename T>
Tensor<T> multiply(const Tensor<T>& t1, const Indices index1,
                   const Tensor<T>& t2, const Indices& index2) {
  Tensor<T> result;
  if (t1.template isType<typename Tensor<T>::Dense>() &&
      t2.template isType<typename Tensor<T>::Dense>()) {
    auto& ref1 = t1.template reference<typename Tensor<T>::Dense>();
    auto& ref2 = t2.template reference<typename Tensor<T>::Dense>();
    result = Tensor<T>(multiply<T>(ref1, index1, ref2, index2));
  } else {
    throw Util::unimplemented_exception(
        "apply binary not implemented for type");
  }

  return result;
}
*/

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
template <typename T>
Tensor<T> multiply(const Tensor<T>& t1, const Indices& indices1,
                   const Tensor<T>& t2, const Indices& indices2) {
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
  return Tensor<T>(result);
}

}  // Tensor

#endif  // TENSOR_TENSOR_BASE_H
