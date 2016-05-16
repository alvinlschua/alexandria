#ifndef AUTOMATIC_DIFFERENTIATION_AD_UNARY_TENSOR_H_
#define AUTOMATIC_DIFFERENTIATION_AD_UNARY_TENSOR_H_

#include <memory>
#include <string>

#include "automatic_differentiation/ad_const_tensor.h"
#include "automatic_differentiation/ad_expression_tensor.h"
#include "automatic_differentiation/ad_tensor.h"
#include "automatic_differentiation/ad_var_tensor.h"
#include "util/clonable.h"

namespace Alexandria {

template <typename T>
class AD<T>::Unary : public Expression {
 public:
  using VarValues = typename AD<T>::VarValues;

  explicit Unary(const AD<T>& term) : term_(term) {}
  Unary(const Unary&) = default;
  Unary& operator=(const Unary&) = default;

  virtual ~Unary() {}

  const AD& term() const { return term_; }
  const Shape& shapeTerm() const { return shapeTermImpl(); }

 private:
  // Value evaluation of the function itself.
  virtual T f(const T& value) const = 0;

  // Differential of the function with respect to the expression.
  virtual AD<T> dF() const = 0;

  // Differentiate with respect to var.
  AD<T> differentiateImpl(const AD<T>& var) const final;

  // Evaluate the expression.
  AD<T> evaluateAtImpl(const VarValues& varValues) const final;

  // Simplify the expression.
  AD<T> simplifyImpl() const override;

  // Shape of function argument.
  virtual const Shape& shapeTermImpl() const = 0;

  AD<T> term_;
};

template <typename T>
AD<T> AD<T>::Unary::differentiateImpl(const AD<T>& var) const {
  // TODO(alvin) Only reverse mode at the moment. Consider implementing forward
  // mode.
  Indices indices1(this->shape().nDimensions() + shapeTerm().nDimensions());
  Indices indices2(shapeTerm().nDimensions() + var.shape().nDimensions());

  const auto resultDimensions = this->shape().nDimensions();
  for (auto index = 0ul; index < indices1.size(); ++index) {
    indices1[index] = static_cast<int>(
        index < resultDimensions ? index : resultDimensions - index - 1);
  }

  for (auto index = 0ul; index < indices2.size(); ++index) {
    indices2[index] = static_cast<int>(index < shapeTerm().nDimensions()
                                           ? -index - 1
                                           : index - shapeTerm().nDimensions() +
                                                 resultDimensions);
  }

  auto result = multiply(dF(), indices1, term().differentiate(var), indices2);
  return result.simplify();
}

template <typename T>
AD<T> AD<T>::Unary::evaluateAtImpl(const VarValues& varValues) const {
  using Const = typename AD<T>::Const;
  auto term = !this->term().template isType<Const>()
                  ? this->term().evaluateAt(varValues)
                  : this->term();
  term = term.simplify();

  if (term.template isType<Const>()) {
    return AD<T>(f(term.template reference<Const>().value()));
  }

  auto ptr = this->clone();
  dynamic_cast<Unary*>(ptr.get())->term_ = term;

  return AD<T>(std::move(ptr));
}

template <typename T>
AD<T> AD<T>::Unary::simplifyImpl() const {
  return term().template isType<typename AD<T>::Const>()
             ? AD<T>(f(value(term())))
             : AD<T>(this->clone());
}

template <typename T>
class UnaryMinus : public AD<T>::Unary {
 public:
  using VarValues = typename AD<T>::VarValues;

  UnaryMinus(const UnaryMinus&) = default;
  UnaryMinus& operator=(const UnaryMinus&) = default;
  virtual ~UnaryMinus() {}

  static AD<T> makeAD(const AD<T>& ad) { return AD<T>(UnaryMinus(ad).clone()); }

 private:
  explicit UnaryMinus(const AD<T>& ad) : AD<T>::Unary(ad) {}

  T f(const T& value) const final { return -value; }
  AD<T> dF() const final {
    return AD<T>(-T::sparseEye(combineShapes(this->shape(), this->shape())));
  }

  const Shape& shapeTermImpl() const final { return this->term().shape(); }
  const Shape& shapeImpl() const final { return this->term().shape(); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<UnaryMinus>(*this);
  }

  std::string expressionImpl() const {
    return this->term().template isType<typename AD<T>::Const>() ||
                   this->term().template isType<typename AD<T>::Var>()
               ? "-" + this->term().expression()
               : "-(" + this->term().expression() + ")";
  }

  AD<T> simplifyImpl() const final;
};

template <typename T>
class Reshape : public AD<T>::Unary {
 public:
  using VarValues = typename AD<T>::VarValues;
  using Sparse = typename T::Sparse;

  Reshape(const Reshape&) = default;
  Reshape& operator=(const Reshape&) = default;
  virtual ~Reshape() {}

  static AD<T> makeAD(const AD<T>& ad, const Shape& resultShape) {
    return AD<T>(Reshape(ad, resultShape).clone());
  }

 private:
  explicit Reshape(const AD<T>& ad, const Shape& resultShape)
      : AD<T>::Unary(ad),
        resultShape_(resultShape),
        permute_(T::sparse(combineShapes(resultShape, ad.shape()))) {
    if (nElements(this->shape()) != nElements(this->shapeTerm())) {
      throw std::invalid_argument(
          "cannot reshape into a shape of different dimensions");
    }

    auto size = nElements(resultShape);
    auto resultAccesser = Accesser(&resultShape);
    auto termAccesser = Accesser(&ad.shape());
    Address address(permute_.shape().nDimensions());
    for (auto index = 0ul; index != size; ++index) {
      auto resultAddress = resultAccesser.address(index);
      auto termAddress = termAccesser.address(index);
      auto iter = std::copy(resultAddress.cbegin(), resultAddress.cend(),
                            address.begin());
      std::copy(termAddress.cbegin(), termAddress.cend(), iter);
      permute_[address] = 1;
    }

    indicesPermute_ = Indices(permute_.shape().nDimensions());
    for (auto index = 0ul; index < indicesPermute_.size(); ++index) {
      indicesPermute_[index] =
          static_cast<int>(index < resultShape.nDimensions()
                               ? index
                               : resultShape.nDimensions() - index - 1);
    }

    indices_ = Indices(ad.shape().nDimensions());
    for (auto index = 0ul; index < indices_.size(); ++index) {
      indices_[index] = static_cast<int>(-index - 1);
    }
  }

  T f(const T& value) const final {
    return multiply(permute_, indicesPermute_, value, indices_);
  }
  AD<T> dF() const final { return AD<T>(T(permute_)); }

  const Shape& shapeTermImpl() const final { return this->term().shape(); }
  const Shape& shapeImpl() const final { return resultShape_; }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Reshape>(*this);
  }

  std::string expressionImpl() const { return "reshape"; }

  AD<T> simplifyImpl() const final { return AD<T>(this->clone()); }

  Shape resultShape_;
  T permute_;
  Indices indicesPermute_;
  Indices indices_;
};

template <typename T>
class Diagonal : public AD<T>::Unary {
 public:
  using VarValues = typename AD<T>::VarValues;
  using Sparse = typename T::Sparse;

  Diagonal(const Diagonal&) = default;
  Diagonal& operator=(const Diagonal&) = default;
  virtual ~Diagonal() {}

  static AD<T> makeAD(const AD<T>& ad) { return AD<T>(Diagonal(ad).clone()); }

 private:
  explicit Diagonal(const AD<T>& ad)
      : AD<T>::Unary(ad),
        resultShape_(combineShapes(ad.shape(), ad.shape())),
        indicesEye_(resultShape_.nDimensions()),
        indices_(ad.shape().nDimensions()) {
    iota(indicesEye_.begin(), indicesEye_.end(), 0);
    iota(indices_.begin(), indices_.end(), indices_.size());
  }

  T f(const T& value) const final {
    return multiply(T::sparseEye(this->shape()), indicesEye_, value, indices_);
  }
  AD<T> dF() const final {
    auto indices = Indices(this->shape().nDimensions());
    iota(indices.begin(), indices.end(), indices_.size());
    auto eye = T::sparseEye(this->shape());
    return AD<T>(multiply(eye, indicesEye_, eye, indices));
  }

  const Shape& shapeTermImpl() const final { return this->term().shape(); }
  const Shape& shapeImpl() const final { return resultShape_; }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Diagonal>(*this);
  }

  std::string expressionImpl() const { return "diagonal"; }

  AD<T> simplifyImpl() const final { return AD<T>(this->clone()); }

  Shape resultShape_;
  Indices indicesEye_;
  Indices indices_;
};

template <typename T>
class Sigmoid : public AD<T>::Unary {
 public:
  using VarValues = typename AD<T>::VarValues;
  using Sparse = typename T::Sparse;

  Sigmoid(const Sigmoid&) = default;
  Sigmoid& operator=(const Sigmoid&) = default;
  virtual ~Sigmoid() {}

  static AD<T> makeAD(const AD<T>& ad) { return AD<T>(Sigmoid(ad).clone()); }

 private:
  explicit Sigmoid(const AD<T>& ad) : AD<T>::Unary(ad) {}

  T f(const T& value) const final {
    return apply<typename T::ValueType>(
        value, [](typename T::ValueType x) { return 1.0 / (1.0 + exp(-x)); });
  }
  AD<T> dF() const final {
    Indices diagIndices(2 * this->shape().nDimensions());
    Indices indices(this->shape().nDimensions());
    iota(diagIndices.begin(), diagIndices.end(), 0);
    iota(indices.begin(), indices.end(), 0);
    return AD<T>(
        multiply(diagonal(T::ones(this->shapeTerm()) - sigmoid(this->term())),
                 diagIndices, sigmoid(this->term()), indices));
  }

  const Shape& shapeTermImpl() const final { return this->term().shape(); }
  const Shape& shapeImpl() const final { return this->shapeTerm(); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Sigmoid>(*this);
  }

  std::string expressionImpl() const { return "reshape"; }

  AD<T> simplifyImpl() const final { return AD<T>(this->clone()); }
};

// Unary Minus
template <typename T>
AD<T> UnaryMinus<T>::simplifyImpl() const {
  AD<T> result;
  if (this->term().template isType<typename AD<T>::Const>()) {
    result = AD<T>(f(value(this->term())));
  } else if (this->term().template isType<UnaryMinus>()) {
    result = this->term().template reference<UnaryMinus>().term();
  } else {
    result = AD<T>(this->clone());
  }
  return result;
}

template <typename T>
AD<T> operator-(const AD<T>& ad) {
  return UnaryMinus<T>::makeAD(ad);
}

template <typename T>
AD<T> reshape(const AD<T>& ad, const Shape& shape) {
  return Reshape<T>::makeAD(ad, shape);
}

template <typename T>
AD<T> sigmoid(const AD<T>& ad) {
  return Sigmoid<T>::makeAD(ad);
}

template <typename T>
AD<T> diagonal(const AD<T>& ad) {
  return Diagonal<T>::makeAD(ad);
}

}  // namespace Alexandria

#endif  // AUTOMATIC_DIFFERENTIATION_AD_UNARY_TENSOR_H_
