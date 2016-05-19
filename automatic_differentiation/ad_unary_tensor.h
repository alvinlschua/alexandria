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

  auto term = this->term().template isType<Const>()
                  ? this->term()
                  : this->term().evaluateAt(varValues);

  term = term.simplify();

  if (term.template isType<Const>()) {
    return AD<T>(f(value(term)));
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
    return AD<T>(
        T::constDiagonal(combineShapes(this->shape(), this->shape()), -1));
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

// Unary Minus
template <typename T>
AD<T> UnaryMinus<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  auto term = this->term().simplify();

  if (term.template isType<Const>()) {
    return AD<T>(f(value(term)));
  }

  if (term.template isType<UnaryMinus>()) {
    return term.template reference<UnaryMinus>().term();
  }

  return -term;
}

template <typename T>
AD<T> operator-(const AD<T>& ad) {
  return UnaryMinus<T>::makeAD(ad);
}

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
      permute_.set(address, 1);
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
    using Dense = typename T::Dense;
    using Sparse = typename T::Sparse;
    if (value.template isType<Dense>()) {
      auto values = value.template reference<Dense>();
      return T(Dense(this->shape(), values.data()));
    } else if (value.template isType<Sparse>()) {
      auto values = value.template reference<Sparse>();
      return T(Sparse(this->shape(), values.data()));
    } else {
      throw unimplemented_exception("unknown tensor type");
    }
  }
  AD<T> dF() const final { return AD<T>(T(permute_)); }

  const Shape& shapeTermImpl() const final { return this->term().shape(); }
  const Shape& shapeImpl() const final { return resultShape_; }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Reshape>(*this);
  }

  std::string expressionImpl() const { return "reshape"; }

  AD<T> simplifyImpl() const final {
    using Const = typename AD<T>::Const;
    auto term = this->term().simplify();

    if (term.template isType<Const>()) {
      return AD<T>(f(value(term)));
    }

    return reshape(term, this->shape());
  }

  Shape resultShape_;
  T permute_;
  Indices indicesPermute_;
  Indices indices_;
};

template <typename T>
AD<T> reshape(const AD<T>& ad, const Shape& shape) {
  return Reshape<T>::makeAD(ad, shape);
}

// f(x) is restricted to f_i(x_i)
template <typename T>
class SeparableFunction : public AD<T>::Unary {
 public:
  using VarValues = typename AD<T>::VarValues;
  using Sparse = typename T::Sparse;

  SeparableFunction(const AD<T>& ad)
      : AD<T>::Unary(ad),
        eyeIndices_(2ul * ad.shape().nDimensions()),
        indices_(ad.shape().nDimensions()),
        eye_(T::sparseEye(combineShapes(ad.shape(), ad.shape()))) {
    iota(eyeIndices_.begin(), eyeIndices_.end(), 0);
    iota(indices_.begin(), indices_.end(), 0);
  }
  SeparableFunction(const SeparableFunction&) = default;
  SeparableFunction& operator=(const SeparableFunction&) = default;
  virtual ~SeparableFunction() {}

 protected:
  const Indices& indices() const { return indices_; }

 private:
  virtual AD<T> dFDiagonal() const = 0;

  AD<T> dF() const final {
    return multiply(eye_, eyeIndices_, dFDiagonal(), indices_);
  }

  const Shape& shapeTermImpl() const final { return this->term().shape(); }
  const Shape& shapeImpl() const final { return this->shapeTerm(); }

  Indices eyeIndices_;
  Indices indices_;
  AD<T> eye_;
};

/*
Sigmoid
f_i = 1 / (1 - exp(-x_i))
df_i(x_i) /dx_i = f_i (1 - f_i)
*/
template <typename T>
class Sigmoid : public SeparableFunction<T> {
 public:
  using VarValues = typename AD<T>::VarValues;
  using Sparse = typename T::Sparse;

  Sigmoid(const Sigmoid&) = default;
  Sigmoid& operator=(const Sigmoid&) = default;
  virtual ~Sigmoid() {}

  static AD<T> makeAD(const AD<T>& ad) { return AD<T>(Sigmoid(ad).clone()); }

 private:
  explicit Sigmoid(const AD<T>& ad) : SeparableFunction<T>(ad) {}

  T f(const T& value) const final {
    return apply<typename T::ValueType>(
        value, [](typename T::ValueType x) { return 1.0 / (1.0 + exp(-x)); });
  }
  AD<T> dFDiagonal() const final {
    return multiply(T::ones(this->shapeTerm()) - sigmoid(this->term()),
                    this->indices(), sigmoid(this->term()), this->indices());
  }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Sigmoid>(*this);
  }

  std::string expressionImpl() const {
    return std::string("sigmoid(") + this->term().expression() + ")";
  }

  AD<T> simplifyImpl() const final {
    using Const = typename AD<T>::Const;

    auto term = this->term().simplify();

    if (term.template isType<Const>()) {
      return AD<T>(f(value(term)));
    }

    return sigmoid(term);
  }
};

template <typename T>
AD<T> sigmoid(const AD<T>& ad) {
  return Sigmoid<T>::makeAD(ad);
}

/*
Log
f_i = log(x_i) -- natural log
df_i(x_i) /dx_i = 1 / x_i
*/
template <typename T>
class Log : public SeparableFunction<T> {
 public:
  using VarValues = typename AD<T>::VarValues;
  using Sparse = typename T::Sparse;

  Log(const Log&) = default;
  Log& operator=(const Log&) = default;
  virtual ~Log() {}

  static AD<T> makeAD(const AD<T>& ad) { return AD<T>(Log(ad).clone()); }

 private:
  explicit Log(const AD<T>& ad) : SeparableFunction<T>(ad) {}

  T f(const T& value) const final {
    return apply<typename T::ValueType>(
        value, [](typename T::ValueType x) { return log(x); });
  }
  AD<T> dFDiagonal() const final { return reciprocal(this->term()); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Log>(*this);
  }

  std::string expressionImpl() const {
    return std::string("log(") + this->term().expression() + ")";
  }

  AD<T> simplifyImpl() const final {
    using Const = typename AD<T>::Const;

    auto term = this->term().simplify();

    if (term.template isType<Const>()) {
      return AD<T>(f(value(term)));
    }

    return log(term);
  }
};

template <typename T>
AD<T> log(const AD<T>& ad) {
  return Log<T>::makeAD(ad);
}

/*
reciprocal
f_i = 1 / x_i -- reciprocal
df_i(x_i) /dx_i = - 1 / (x_i * x_i)
*/
template <typename T>
class Reciprocal : public SeparableFunction<T> {
 public:
  using VarValues = typename AD<T>::VarValues;
  using Sparse = typename T::Sparse;

  Reciprocal(const Reciprocal&) = default;
  Reciprocal& operator=(const Reciprocal&) = default;
  virtual ~Reciprocal() {}

  static AD<T> makeAD(const AD<T>& ad) { return AD<T>(Reciprocal(ad).clone()); }

 private:
  explicit Reciprocal(const AD<T>& ad) : SeparableFunction<T>(ad) {}

  T f(const T& value) const final {
    return apply<typename T::ValueType>(
        value, [](typename T::ValueType x) { return 1.0 / x; });
  }
  AD<T> dFDiagonal() const final {
    return -reciprocal(
        multiply(this->term(), this->indices(), this->term(), this->indices()));
  }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Reciprocal>(*this);
  }

  std::string expressionImpl() const {
    return std::string("1 / (") + this->term().expression() + ")";
  }

  AD<T> simplifyImpl() const final {
    using Const = typename AD<T>::Const;

    auto term = this->term().simplify();

    if (term.template isType<Const>()) {
      return AD<T>(f(value(term)));
    }

    if (term.template isType<Reciprocal>()) {
      return term.template reference<Reciprocal>().term();
    }

    return reciprocal(term);
  }
};

template <typename T>
AD<T> reciprocal(const AD<T>& ad) {
  return Reciprocal<T>::makeAD(ad);
}

}  // namespace Alexandria

#endif  // AUTOMATIC_DIFFERENTIATION_AD_UNARY_TENSOR_H_
