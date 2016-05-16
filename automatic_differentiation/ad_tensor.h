#ifndef AUTOMATIC_DIFFERENTIATION_AD_TENSOR_H_
#define AUTOMATIC_DIFFERENTIATION_AD_TENSOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "glog/logging.h"
#include "tensor/tensor.h"
#include "tensor/tensor_sparse.h"
#include "tensor/tensor_dense.h"

// Use this in place of ad when differentiating tensors.
namespace Alexandria {

// A wrapper class that implements a simple for of type erasure.
template <typename T>
class AD {
 public:
  using VarValue = std::pair<AD<T>, T>;
  using VarValues = std::vector<VarValue>;

  class Expression;
  class Const;
  class Param;
  class Var;
  class Unary;
  class Binary;

  using Ptr = std::unique_ptr<Expression>;

  // Default constructor.
  AD() {}

  // Make constant.
  explicit AD(const T& value);

  // Make var.
  explicit AD(const std::string& identifier, const Shape& shape);

  // Make param. Params can be treated as variables, but have an associated set
  // of values.
  explicit AD(const std::string& identifier, const T& value);

  // Construct from a ptr.
  explicit AD(Ptr ptr) : ptr_(std::move(ptr)) {}

  // Copy constructor.
  AD(const AD& ad);

  // Copy assignment.
  AD& operator=(const AD& ad);

  // Produce a VarValue when assigning T to a var.  Not that this is not a
  // constructor, but is implemented for notation.
  AD::VarValue operator=(const T& value);

  // Differentiate with respect to AD::Var.
  AD differentiate(const AD& var) const { return ptr_->differentiate(var); }

  // Evaluate the expression with concrete values for AD::Var.
  AD evaluateAt(const VarValues& varValues) const {
    return ptr_->evaluateAt(varValues);
  }

  // Simplify the expression.
  AD simplify() const { return ptr_->simplify(); }

  // Shape of the result;
  const Shape& shape() const { return ptr_->shape(); }

  // Get the expression as a string.
  std::string expression() const { return ptr_->expression(); }

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
  Ptr ptr_;
};

template <typename T>
AD<T>::AD(const T& value)
    : ptr_(Const::make(value)) {}

template <typename T>
AD<T>::AD(const std::string& identifier, const Shape& shape)
    : ptr_(Var::make(identifier, shape)) {
  if (identifier.size() == 0)
    throw std::invalid_argument("identifier should be specified");

  if (identifier.size() > 8)
    throw std::invalid_argument("identifier should have at most 8 characters");

  std::locale loc;
  if (!std::isalpha(identifier[0], loc))
    throw std::invalid_argument("identifier should start with a letter");
}

template <typename T>
AD<T>::AD(const std::string& identifier, const T& value)
    : ptr_(Param::make(identifier, value)) {
  if (identifier.size() == 0)
    throw std::invalid_argument("identifier should be specified");

  if (identifier.size() > 8)
    throw std::invalid_argument("identifier should have at most 8 characters");

  std::locale loc;
  if (!std::isalpha(identifier[0], loc))
    throw std::invalid_argument("identifier should start with a letter");
}

template <typename T>
AD<T>::AD(const AD& ad) {
  ptr_ = std::move(ad.ptr_->clone());
}

template <typename T>
AD<T>& AD<T>::operator=(const AD& ad) {
  ptr_ = std::move(ad.ptr_->clone());
  return *this;
}

template <typename T>
typename AD<T>::VarValue AD<T>::operator=(const T& value) {
  CHECK(isType<AD::Var>()) << "should only assign T to a Var type";
  return AD::VarValue(*this, value);
}

// Differentiate.
template <typename T>
inline AD<T> D(const AD<T>& expr, const AD<T>& var) {
  return expr.differentiate(var);
}

// TODO(alvin) Make ADVector its own class and provide methods like evaluateAt
// and simplify.
template <typename T>
using ADVector = std::vector<AD<T>>;

template <typename T>
ADVector<T> grad(const AD<T>& expr, const std::vector<AD<T>>& vars);

// Gets the value when ad is a Const.
template <typename T>
T value(const AD<T>& ad) {
  using Const = typename AD<T>::Const;
  using Param = typename AD<T>::Param;

  if (ad.template isType<Const>()) {
    return ad.template reference<Const>().value();
  } else if (ad.template isType<Param>()) {
    return ad.template reference<Param>().value();
  } else {
    throw std::invalid_argument("type is not a const nor a param.");
  }
}

template <typename T>
std::string identifier(const AD<T>& ad) {
  using Var = typename AD<T>::Var;
  using Param = typename AD<T>::Param;

  if (ad.template isType<Var>()) {
    return ad.template reference<Var>().identifier();
  } else if (ad.template isType<Param>()) {
    return ad.template reference<Param>().identifier();
  } else {
    throw std::invalid_argument("type is not a var nor a param.");
  }
}

template <typename T>
T& param(const AD<T>& ad) {
  return ad.template reference<typename AD<T>::Param>().value();
}

template <typename T>
AD<T> D(const AD<T>& expr, const std::vector<AD<T>>& vars) {
  AD<T> result(expr);
  for (const auto& var : vars) {
    result = D(result, var);
  }
  return result;
}

}  // namespace Alexandria

#endif  // AUTOMATIC_DIFFERENTIATION_AD_TENSOR_H_
