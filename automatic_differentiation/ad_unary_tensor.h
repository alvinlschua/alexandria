#ifndef AUTOMATIC_DIFFERENTIATION_AD_UNARY_TENSOR_H_
#define AUTOMATIC_DIFFERENTIATION_AD_UNARY_TENSOR_H_

#include <memory>
#include <string>

#include "automatic_differentiation/ad_const_tensor.h"
#include "automatic_differentiation/ad_expression_tensor.h"
#include "automatic_differentiation/ad_tensor.h"
#include "automatic_differentiation/ad_var_tensor.h"
#include "util/clonable.h"

namespace AutomaticDifferentiation {

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
  //
  using NeuralNet::Indices;
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
  using Shape = typename AD<T>::Shape;

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
  using Shape = typename AD<T>::Shape;
  using Sparse = typename T::Sparse;
  using Indices = typename NeuralNet::Indices;

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
        permute_(T::sparse(NeuralNet::combineShapes(resultShape, ad.shape()))) {
    if (nElements(this->shape()) != nElements(this->shapeTerm())) {
      throw std::invalid_argument(
          "cannot reshape into a shape of different dimensions");
    }

    auto size = nElements(resultShape);
    auto resultAccesser = NeuralNet::Accesser(&resultShape);
    auto termAccesser = NeuralNet::Accesser(&ad.shape());
    NeuralNet::Address address(permute_.shape().nDimensions());
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
    return NeuralNet::multiply(permute_, indicesPermute_, value, indices_);
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
  NeuralNet::Indices indicesPermute_;
  NeuralNet::Indices indices_;
};

/*
template <typename T>
class Sin : public AD<T>::Unary {
 public:
  virtual ~Sin() {}
  Sin(const Sin&) = default;
  Sin& operator=(const Sin&) = default;

  static AD<T> makeAD(const AD<T>& ad) { return AD<T>(Sin(ad).clone()); }

 private:
  explicit Sin(const AD<T>& ad) : AD<T>::Unary(ad) {}

  T f(const T& value) const final { return sin(value); }
  AD<T> dF() const final { return cos(this->term()); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Sin>(*this);
  }

  std::string expressionImpl() const {
    return "sin(" + this->term().expression() + ")";
  }
};

template <typename T>
class Cos : public AD<T>::Unary {
 public:
  virtual ~Cos() {}
  Cos(const Cos&) = default;
  Cos& operator=(const Cos&) = default;

  static AD<T> makeAD(const AD<T>& ad) { return AD<T>(Cos(ad).clone()); }

 private:
  explicit Cos(const AD<T>& ad) : AD<T>::Unary(ad) {}

  T f(const T& value) const final { return cos(value); }
  AD<T> dF() const final { return -sin(this->term()); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Cos>(*this);
  }

  std::string expressionImpl() const {
    return "cos(" + this->term().expression() + ")";
  }
};

template <typename T>
class Exp : public AD<T>::Unary {
 public:
  virtual ~Exp() {}
  Exp(const Exp&) = default;
  Exp& operator=(const Exp&) = default;

  static AD<T> makeAD(const AD<T>& ad) { return AD<T>(Exp(ad).clone()); }

 private:
  explicit Exp(const AD<T>& ad) : AD<T>::Unary(ad) {}

  T f(const T& value) const final { return exp(value); }
  AD<T> dF() const final { return exp(this->term()); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Exp>(*this);
  }

  std::string expressionImpl() const {
    return "exp(" + this->term().expression() + ")";
  }
};
*/
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
AD<T> reshape(const AD<T>& ad, const NeuralNet::Shape& shape) {
  return Reshape<T>::makeAD(ad, shape);
}

/*
// Sin
template <typename T>
AD<T> sin(const AD<T>& ad) {
  return Sin<T>::makeAD(ad);
}

// Cos
template <typename T>
AD<T> cos(const AD<T>& ad) {
  return Cos<T>::makeAD(ad);
}

// Exp
template <typename T>
AD<T> exp(const AD<T>& ad) {
  return Exp<T>::makeAD(ad);
}

// Log
template <typename T>
AD<T> log(const AD<T>& ad) {
  return Log<T>::makeAD(ad);
}
*/

}  // namespace AutomaticDifferentiation

#endif  // AUTOMATIC_DIFFERENTIATION_AD_UNARY_TENSOR_H_
