#ifndef TENSOR_TENSOR_WRAP_H_
#define TENSOR_TENSOR_WRAP_H_

#include <iostream>
#include <vector>

#include "tensor/accesser.h"
#include "tensor/helpers.h"
#include "tensor/shape.h"
#include "util/clonable.h"
#include "util/rng.h"
#include "util/serializable.h"
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
class Tensor : public Serializable {
 public:
  class Base;
  class Dense;
  class Sparse;
  class AddressIterator;

  using Ptr = std::unique_ptr<Base>;
  using ValueType = T;
  using Data1d = std::vector<T>;
  using Data2d = std::vector<std::vector<T>>;
  using Data3d = std::vector<std::vector<std::vector<T>>>;

  virtual ~Tensor() {}

  Tensor() : ptr_(std::make_unique<Dense>()) {}

  Tensor(const Tensor& tensor) : ptr_(tensor.ptr_->clone()) {}
  Tensor& operator=(const Tensor& tensor) {
    ptr_ = std::move(tensor.ptr_->clone());
    return *this;
  }

  explicit Tensor(const Dense& tensor) : ptr_(tensor.clone()) {}
  explicit Tensor(const Sparse& tensor) : ptr_(tensor.clone()) {}

  explicit Tensor(const Data1d& data)
      : ptr_(Dense(Shape({data.size()}), data).clone()) {}
  explicit Tensor(const Data2d& data);
  explicit Tensor(const Data3d& data);

  static Tensor<T> fill(const Shape& shape, T value);
  static Tensor zeros(const Shape& shape);
  static Tensor ones(const Shape& shape);

  // uninitialized dense
  static Tensor dense(const Shape& shape);
  static Tensor sparse(const Shape& shape);
  static Tensor generate(const Shape& shape, std::function<T(Address)> fn);

  template <typename TDistribution>
  static Tensor random(const Shape& shape, const TDistribution& distribution);
  static Tensor random(const Shape& shape);

  // shape
  static Tensor sparseEye(const Shape& shape, T value = 1);

  // Return the number of elements.
  size_t size() const { return ptr_->size(); }

  // Return the shape.
  const Shape& shape() const { return ptr_->shape(); }

  // Access a const element.
  T at(const Address& address) const { return ptr_->at(address); }

  // Access a const element.
  T operator[](const Address& address) const { return ptr_->at(address); }

  // Set the value at the address.
  void set(const Address& address, T value,
           std::function<T(T, T)> fn = [](T /*init*/, T v) { return v; }) {
    return ptr_->set(address, value, fn);
  }

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

  // Address iterator begin and end cycles through non zero values of the
  // tensor.
  AddressIterator cbegin() const { return ptr_->begin(); }
  AddressIterator cend() const { return ptr_->end(); }
  AddressIterator begin() const { return cbegin(); }
  AddressIterator end() const { return cend(); }

 private:
  using ArchiveIn = ArchiveIn;
  using ArchiveOut = ArchiveOut;

  Tensor(Ptr ptr) : ptr_(std::move(ptr)) {}

  void serializeInImpl(ArchiveIn& ar, size_t /*version*/) final;
  void serializeOutImpl(ArchiveOut& ar) const final;
  size_t serializeOutVersionImpl() const final { return 0ul; }

  Ptr ptr_;
};

template <typename T>
bool operator==(const Tensor<T>& t1, const Tensor<T>& t2) {
  using Dense = typename Tensor<T>::Dense;

  auto result = true;
  if (t1.template isType<Dense>() && t2.template isType<Dense>()) {
    auto x = t1.template reference<Dense>();
    auto y = t2.template reference<Dense>();
    result = x.shape() == y.shape() &&
             std::equal(x.data().cbegin(), x.data().cend(), y.data().cbegin(),
                        [](T x1, T x2) { return almostEqual(x1, x2); });
  } else {
    if (t1.shape() != t2.shape()) return false;
    Address address(t1.shape().nDimensions(), 0);
    auto size = nElements(t1.shape());
    for (auto index = 0ul; index < size;
         ++index, address = increment(std::move(address), t1.shape())) {
      if (!almostEqual(t1[address], t2[address])) {
        result = false;
        break;
      }
    }
  }

  return result;
}

template <typename T>
bool operator!=(const Tensor<T>& t1, const Tensor<T>& t2) {
  return !(t1 == t2);
}

template <typename T>
Tensor<T> Tensor<T>::generate(const Shape& shape,
                              std::function<T(Address)> fn) {
  using Dense = typename Tensor<T>::Dense;
  auto result = Tensor(Dense(shape));
  auto& dense = result.template reference<Dense>();

  Address address(shape.nDimensions(), 0ul);
  for (auto& value : dense.data()) {
    value = fn(address);
    address = increment(std::move(address), result.shape());
  }

  return result;
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
Tensor<T>::Tensor(const Data2d& data) {
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
Tensor<T>::Tensor(const Data3d& data) {
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

// Makes an eye with shape x shape, so that e_ij,lm = delta_il delta_jm
template <typename T>
Tensor<T> Tensor<T>::sparseEye(const Shape& shape, T value) {
  if (!isEyeShape(shape)) {
    throw std::invalid_argument("shape not suitable for eye");
  }

  std::vector<size_t> dims(shape.begin(),
                           shape.begin() + shape.nDimensions() / 2);
  Shape parseShape = Shape(dims);

  Address address(parseShape.nDimensions(), 0);
  auto size = nElements(shape);

  auto result = Tensor(Sparse(shape));
  auto resultAddress = Address(shape.nDimensions());
  for (auto index = 0ul; index < size; ++index) {
    auto iter =
        std::copy(address.cbegin(), address.cend(), resultAddress.begin());
    std::copy(address.cbegin(), address.cend(), iter);

    result.set(resultAddress, value);
    address = increment(std::move(address), shape);
  }

  return result;
}

template <typename T>
void Tensor<T>::serializeInImpl(ArchiveIn& ar, size_t /*version*/) {
  bool isDense = false;
  ar % isDense;
  if (isDense) {
    ptr_ = std::make_unique<Dense>();
    ar % (*ptr_);
  } else {
    ptr_ = std::make_unique<Sparse>();
    ar % (*ptr_);
  }
}

template <typename T>
void Tensor<T>::serializeOutImpl(ArchiveOut& ar) const {
  bool isDense = this->isType<Dense>();
  ar % isDense % (*ptr_);
}

template <typename T>
Tensor<T> Tensor<T>::fill(const Shape& shape, T value) {
  auto size = nElements(shape);
  auto data = typename Dense::Data(size, value);

  return Tensor<T>(Dense(shape, std::move(data)));
}

template <typename T>
Tensor<T> Tensor<T>::ones(const Shape& shape) {
  return fill(shape, 1);
}

template <typename T>
Tensor<T> Tensor<T>::zeros(const Shape& shape) {
  return fill(shape, 0);
}

// Make a random tensor. TDistribution should be a RandomNumberDistribution
// concept.
template <typename T>
template <typename TDistribution>
Tensor<T> Tensor<T>::random(const Shape& shape,
                            const TDistribution& distribution) {
  auto data = rng().generate(distribution, nElements(shape));
  return Tensor<T>(Dense(shape, std::move(data)));
}

// Make a uniform(-1, 1) random tensor.
template <typename T>
Tensor<T> Tensor<T>::random(const Shape& shape) {
  return random(shape, std::uniform_real_distribution<T>(-1, 1));
}

// Note: Moving apply functionality is worth it only if it is treated as a
// member function.
template <typename T>
Tensor<T> apply(Tensor<T> t, std::function<T(T)> fn) {
  using Dense = typename Tensor<T>::Dense;
  using Sparse = typename Tensor<T>::Sparse;

  if (t.template isType<Dense>()) {
    auto& temp = t.template reference<Dense>();
    std::transform(temp.data().cbegin(), temp.data().cend(),
                   temp.data().begin(), fn);
  } else if (t.template isType<Sparse>()) {
    if (!almostEqual(fn(0), 0)) {
      throw std::invalid_argument(
          "apply function for sparse tensors must obey f(0) == 0");
    }
    auto& sparse = t.template reference<Sparse>();
    for (auto& value : sparse.data()) {
      value.second = fn(value.second);
    }
  } else {
    CHECK(false) << "cannot apply to this tensor type" << std::endl;
  }

  return t;
}

// Note: Moving apply functionality into derived classes is worth it only if
// it is treated as a member function. Assumes result derived time is the
// same
// as t1
template <typename T>
Tensor<T> apply(Tensor<T> t1, const Tensor<T>& t2, std::function<T(T, T)> fn) {
  using Dense = typename Tensor<T>::Dense;

  if (t1.shape() != t2.shape()) {
    throw std::invalid_argument("shapes are not the same");
  }

  if (t1.template isType<Dense>() && t2.template isType<Dense>()) {
    auto& temp1 = t1.template reference<Dense>();
    auto& temp2 = t2.template reference<Dense>();
    std::transform(temp1.data().cbegin(), temp1.data().cend(),
                   temp2.data().cbegin(), temp1.data().begin(), fn);
  } else if (t1.template isType<Dense>()) {
    for (const auto& address_value : t1) {
      t1.set(address_value.first,
             fn(address_value.second, t2.at(address_value.first)));
    }
  } else {
    Address address(t1.shape().nDimensions(), 0);
    auto size = nElements(t1.shape());
    for (auto index = 0ul; index < size;
         ++index, address = increment(std::move(address), t1.shape())) {
      t1.set(address, fn(t1.at(address), t2.at(address)));
    }
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
Tensor<T> multiply(const Tensor<T>& t1, const Indices& indices1,
                   const Tensor<T>& t2, const Indices& indices2) {
  using namespace Alexandria;
  using namespace std;

  Shape result_shape;
  Shape common_shape;

  tie(result_shape, common_shape) =
      multiplyShapes(t1.shape(), indices1, t2.shape(), indices2);
  auto result_accesser = Accesser(&result_shape);
  auto common_accesser = Accesser(&common_shape);

  auto result_address_iter = result_accesser.cbegin();

  auto result = typename Tensor<T>::Dense(
      result_shape, std::vector<T>(nElements(result_shape), 0));
  for (auto iter = result.dataBegin(); iter != result.dataEnd(); ++iter) {
    auto& element = *iter;
    Address address1(t1.shape().nDimensions());
    Address address2(t2.shape().nDimensions());

    gather(indices1.cbegin(), indices1.cend(),
result_address_iter->cbegin(),
           address1.begin(),
           [](int index) { return index >= 0 ? index : invalid_index; });

    gather(indices2.cbegin(), indices2.cend(),
result_address_iter->cbegin(),
           address2.begin(),
           [](int index) { return index >= 0 ? index : invalid_index; });

    if (common_shape.nDimensions() > 0) {
      for (const auto& common_address : common_accesser) {
        gather(indices1.cbegin(), indices1.cend(), common_address.cbegin(),
               address1.begin(),
               [](int idx) { return idx < 0 ? -idx - 1 : invalid_index; });

        gather(indices2.cbegin(), indices2.cend(), common_address.cbegin(),
               address2.begin(),
               [](int idx) { return idx < 0 ? -idx - 1 : invalid_index; });

        element += t1[address1] * t2[address2];
      }
    } else {
      element = t1[address1] * t2[address2];
    }
    ++result_address_iter;
  }
  return Tensor<T>(result);
}
*/

template <typename T>
Tensor<T> multiply(const Tensor<T>& t1, const Indices& indices1,
                   const Tensor<T>& t2, const Indices& indices2) {
  using namespace Alexandria;
  using namespace std;

  Shape result_shape;

  tie(result_shape, std::ignore) =
      multiplyShapes(t1.shape(), indices1, t2.shape(), indices2);

  auto result = Tensor<T>::zeros(result_shape);
  auto result_address = Address(result_shape.nDimensions());

  auto common_indices1 = Indices(indices1.size());
  auto common_indices2 = Indices(indices2.size());

  // grab common indices.
  transform(indices1.cbegin(), indices1.cend(), common_indices1.begin(),
            [&indices2](auto index) {
              return std::find(indices2.cbegin(), indices2.cend(), index) !=
                             indices2.cend()
                         ? index
                         : invalid_index;
            });

  transform(indices2.cbegin(), indices2.cend(), common_indices2.begin(),
            [&indices1](auto index) {
              return std::find(indices1.cbegin(), indices1.cend(), index) !=
                             indices1.cend()
                         ? index
                         : invalid_index;
            });

  // reindex in ascending order
  Indices reindex;
  copy_if(common_indices1.cbegin(), common_indices1.cend(),
          back_inserter(reindex),
          [](auto index) { return index != invalid_index; });

  sort(reindex.begin(), reindex.end());

  transform(common_indices1.cbegin(), common_indices1.cend(),
            common_indices1.begin(), [&reindex](auto index) {
              return index == invalid_index
                         ? invalid_index
                         : find(reindex.cbegin(), reindex.cend(), index) -
                               reindex.cbegin();
            });
  transform(common_indices2.cbegin(), common_indices2.cend(),
            common_indices2.begin(), [&reindex](auto index) {
              return index == invalid_index
                         ? invalid_index
                         : find(reindex.cbegin(), reindex.cend(), index) -
                               reindex.cbegin();
            });

  auto count = static_cast<size_t>(
      count_if(common_indices1.cbegin(), common_indices1.cend(),
               [](auto index) { return index != invalid_index; }));

  auto common_address1 = Address(count);
  auto common_address2 = Address(count);

  for (const auto& address_value1 : t1) {
    scatter(common_indices1.cbegin(), common_indices1.cend(),
            address_value1.first.cbegin(), common_address1.begin());
    scatter(indices1.cbegin(), indices1.cend(), address_value1.first.cbegin(),
            result_address.begin(),
            [](int index) { return index >= 0 ? index : invalid_index; });
    for (const auto& address_value2 : t2) {
      scatter(common_indices2.cbegin(), common_indices2.cend(),
              address_value2.first.cbegin(), common_address2.begin());

      if (common_address1 != common_address2) continue;

      scatter(indices2.cbegin(), indices2.cend(), address_value2.first.cbegin(),
              result_address.begin(),
              [](int index) { return index >= 0 ? index : invalid_index; });

      result.set(result_address, address_value1.second * address_value2.second,
                 [](T init, T value) { return init + value; });
    }
  }
  return Tensor<T>(result);
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& t) {
  const auto& shape = t.shape();
  const auto size = nElements(shape);

  out << shape;
  if (size > 30) {
    out << "{ " << size << " elements }";
  } else {
    out << "{ ";
    Address address(t.shape().nDimensions(), 0);
    auto idx_size = nElements(t.shape());
    for (auto index = 0ul; index < idx_size; ++index) {
      out << t[address] << " ";
      address = increment(std::move(address), t.shape());
    }
    out << "}";
  }

  return out;
}

}  // Tensor

#endif  // TENSOR_TENSOR_WRAP_H_
