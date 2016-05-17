#ifndef AUTOMATIC_DIFFERENTIATION_AD_BINARY_TENSOR_H_
#define AUTOMATIC_DIFFERENTIATION_AD_BINARY_TENSOR_H_

#include <locale>
#include <memory>
#include <string>

#include "automatic_differentiation/ad_const_tensor.h"
#include "automatic_differentiation/ad_expression_tensor.h"
#include "automatic_differentiation/ad_tensor.h"
#include "automatic_differentiation/ad_var_tensor.h"
#include "util/clonable.h"
#include "util/util.h"

namespace Alexandria {

template <typename T>
class AD<T>::Binary : public Expression {
 public:
  using VarValues = AD<T>::VarValues;

  Binary(const AD<T>& term1, const AD<T>& term2)
      : term1_(term1), term2_(term2) {}
  Binary(const Binary&) = default;
  Binary& operator=(const Binary&) = default;

  virtual ~Binary() {}

  const AD<T>& term1() const { return term1_; }
  const AD<T>& term2() const { return term2_; }
  const Shape& shapeTerm1() const { return shapeTerm1Impl(); }
  const Shape& shapeTerm2() const { return shapeTerm2Impl(); }

 private:
  // TODO(alvin) Considering adding fMove

  // Evaluate the unary function.
  virtual T f(const T& value1, const T& value2) const = 0;

  // Differential of the function with respect to the first expression.
  virtual AD<T> dF1() const = 0;

  // Differential of the function with respect to the second expression.
  virtual AD<T> dF2() const = 0;

  // Differentiate with respect to var.
  AD<T> differentiateImpl(const AD<T>& var) const;

  // Evaluate the expression.
  AD<T> evaluateAtImpl(const VarValues& varValues) const final;

  virtual const Shape& shapeTerm1Impl() const = 0;
  virtual const Shape& shapeTerm2Impl() const = 0;

  AD<T> term1_;
  AD<T> term2_;
};

template <typename T>
AD<T> AD<T>::Binary::differentiateImpl(const AD<T>& var) const {
  // TODO(alvin) Only reverse mode at the moment. Consider implementing forward
  // mode.

  Indices indices11(this->shape().nDimensions() + shapeTerm1().nDimensions());
  Indices indices12(shapeTerm1().nDimensions() + var.shape().nDimensions());
  Indices indices21(this->shape().nDimensions() + shapeTerm2().nDimensions());
  Indices indices22(shapeTerm2().nDimensions() + var.shape().nDimensions());

  // dF1 has shape {resultShape, term1Shape} d/dvar has shape {term1Shape, var}
  const auto resultDimensions = this->shape().nDimensions();
  for (auto index = 0ul; index < indices11.size(); ++index) {
    indices11[index] = static_cast<int>(
        index < resultDimensions ? index : resultDimensions - index - 1);
  }

  for (auto index = 0ul; index < indices12.size(); ++index) {
    indices12[index] = static_cast<int>(
        index < shapeTerm1().nDimensions()
            ? -index - 1
            : index - shapeTerm1().nDimensions() + resultDimensions);
  }

  for (auto index = 0ul; index < indices21.size(); ++index) {
    indices21[index] = static_cast<int>(
        index < resultDimensions ? index : resultDimensions - index - 1);
  }

  for (auto index = 0ul; index < indices22.size(); ++index) {
    indices22[index] = static_cast<int>(
        index < shapeTerm2().nDimensions()
            ? -index - 1
            : index - shapeTerm2().nDimensions() + resultDimensions);
  }

  auto result =
      multiply(dF1(), indices11, term1().differentiate(var), indices12) +
      multiply(dF2(), indices21, term2().differentiate(var), indices22);

  return result.simplify();
}

template <typename T>
AD<T> AD<T>::Binary::evaluateAtImpl(const VarValues& varValues) const {
  using Const = typename AD<T>::Const;

  auto term1 = this->term1().template isType<Const>()
                   ? this->term1()
                   : this->term1().evaluateAt(varValues);
  auto term2 = this->term2().template isType<Const>()
                   ? this->term2()
                   : this->term2().evaluateAt(varValues);

  term1 = term1.simplify();
  term2 = term2.simplify();

  if (term1.template isType<Const>() && term2.template isType<Const>()) {
    return AD<T>(f(value(term1), value(term2)));
  }

  auto ptr = this->clone();
  dynamic_cast<Binary*>(ptr.get())->term1_ = term1;
  dynamic_cast<Binary*>(ptr.get())->term2_ = term2;

  return AD<T>(std::move(ptr)).simplify();
}

template <typename T>
class Plus : public AD<T>::Binary {
 public:
  Plus(const Plus&) = default;
  Plus& operator=(const Plus&) = default;
  virtual ~Plus() {}

  static AD<T> makeAD(const AD<T>& term1, const AD<T>& term2) {
    return AD<T>(Plus(term1, term2).clone());
  }

 private:
  Plus(const AD<T>& term1, const AD<T>& term2) : AD<T>::Binary(term1, term2) {
    if (term1.shape() != term2.shape()) {
      throw std::invalid_argument("tensor shapes do not match");
    }
  }

  T f(const T& value1, const T& value2) const final { return value1 + value2; }
  AD<T> dF1() const final {
    return AD<T>(T::sparseEye(combineShapes(this->shape(), this->shape())));
  }
  AD<T> dF2() const final {
    return AD<T>(T::sparseEye(combineShapes(this->shape(), this->shape())));
  }
  const Shape& shapeTerm1Impl() const final { return this->term1().shape(); }
  const Shape& shapeTerm2Impl() const final { return this->term2().shape(); }
  const Shape& shapeImpl() const final { return this->term1().shape(); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Plus>(*this);
  }

  std::string expressionImpl() const {
    return "(" + this->term1().expression() + " + " +
           this->term2().expression() + ")";
  }

  AD<T> simplifyImpl() const final;
};

template <typename T>
AD<T> Plus<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  auto term1 = this->term1().simplify();
  auto term2 = this->term2().simplify();

  if (term1.template isType<Const>() && term2.template isType<Const>()) {
    return AD<T>(f(value(term1), value(term2)));
  }

  // Convert to Const + Expression if possible
  if (term2.template isType<Const>()) {
    std::swap(term1, term2);
  }

  // 0 + Expression -> Expression
  if (term1.template isType<Const>() &&
      value(term1) == T::zeros(term1.shape())) {
    return term2;
  }

  return term1 + term2;
}

template <typename T>
AD<T> operator+(const AD<T>& term1, const AD<T>& term2) {
  return Plus<T>::makeAD(term1, term2);
}

template <typename T>
AD<T> operator+(const T& term1, const AD<T>& term2) {
  return Plus<T>::makeAD(AD<T>(term1), term2);
}

template <typename T>
AD<T> operator+(const AD<T>& term1, const T& term2) {
  return Plus<T>::makeAD(term1, AD<T>(term2));
}

template <typename T>
class Minus : public AD<T>::Binary {
 public:
  Minus(const Minus&) = default;
  Minus& operator=(const Minus&) = default;
  virtual ~Minus() {}

  static AD<T> makeAD(const AD<T>& term1, const AD<T>& term2) {
    return AD<T>(Minus(term1, term2).clone());
  }

 private:
  Minus(const AD<T>& term1, const AD<T>& term2) : AD<T>::Binary(term1, term2) {
    if (term1.shape() != term2.shape()) {
      throw std::invalid_argument("tensor shapes do not match");
    }
  }

  T f(const T& value1, const T& value2) const final { return value1 - value2; }
  AD<T> dF1() const final {
    return AD<T>(T::sparseEye(combineShapes(this->shape(), this->shape())));
  }
  AD<T> dF2() const final {
    return AD<T>(-1.0 *
                 T::sparseEye(combineShapes(this->shape(), this->shape())));
  }
  const Shape& shapeTerm1Impl() const final { return this->term1().shape(); }
  const Shape& shapeTerm2Impl() const final { return this->term2().shape(); }
  const Shape& shapeImpl() const final { return this->term1().shape(); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Minus>(*this);
  }

  std::string expressionImpl() const {
    return "(" + this->term1().expression() + " - " +
           this->term2().expression() + ")";
  }

  AD<T> simplifyImpl() const final;
};

// Minus
template <typename T>
AD<T> Minus<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  auto term1 = this->term1().simplify();
  auto term2 = this->term2().simplify();

  if (term1.template isType<Const>() && term2.template isType<Const>()) {
    return AD<T>(f(value(term1), value(term2)));
  }

  if (term1.template isType<Const>() &&
      value(term1) == T::zeros(term1.shape())) {
    return -term2;
  }

  if (term2.template isType<Const>() &&
      value(term2) == T::zeros(term2.shape())) {
    return term1;
  }

  return term1 - term2;
}

template <typename T>
AD<T> operator-(const AD<T>& term1, const AD<T>& term2) {
  return Minus<T>::makeAD(term1, term2);
}

template <typename T>
AD<T> operator-(const T& value, const AD<T>& term2) {
  return Minus<T>::makeAD(AD<T>(value), term2);
}

template <typename T>
AD<T> operator-(const AD<T>& term1, const T& value) {
  return Minus<T>::makeAD(term1, AD<T>(value));
}

template <typename T>
class Multiply : public AD<T>::Binary {
 public:
  Multiply(const Multiply&) = default;
  Multiply& operator=(const Multiply&) = default;
  virtual ~Multiply() {}

  static AD<T> makeAD(const AD<T>& term1, const Indices& indices1,
                      const AD<T>& term2, const Indices& indices2) {
    return AD<T>(Multiply(term1, indices1, term2, indices2).clone());
  }

  const Indices& indices1() const { return indices1_; }
  const Indices& indices2() const { return indices2_; }

 private:
  Multiply(const AD<T>& term1, const Indices& indices1, const AD<T>& term2,
           const Indices& indices2)
      : AD<T>::Binary(term1, term2), indices1_(indices1), indices2_(indices2) {
    std::tie(resultShape_, std::ignore) =
        multiplyShapes(term1.shape(), indices1, term2.shape(), indices2);
  }

  T f(const T& value1, const T& value2) const final {
    // Multiply mimics the return type of value1.
    // We return sparse if possible.
    return (value1.template isType<typename T::Dense>() &&
            value2.template isType<typename T::Sparse>())
               ? multiply(value2, indices2(), value1, indices1())
               : multiply(value1, indices1(), value2, indices2());
  }
  AD<T> dF1() const final {
    auto eyeIndex = Indices(2 * indices1().size());
    auto iter =
        std::copy(indices1().cbegin(), indices1().cend(), eyeIndex.begin());
    std::iota(iter, eyeIndex.end(), this->shape().nDimensions());
    const auto eye = T::sparseEye(
        combineShapes(this->term1().shape(), this->term1().shape()));
    return multiply(AD<T>(eye), eyeIndex, this->term2(), indices2());
  }
  AD<T> dF2() const final {
    auto eyeIndex = Indices(2 * indices2().size());
    auto iter =
        std::copy(indices2().cbegin(), indices2().cend(), eyeIndex.begin());
    std::iota(iter, eyeIndex.end(), this->shape().nDimensions());
    const auto eye = T::sparseEye(
        combineShapes(this->term2().shape(), this->term2().shape()));
    return multiply(this->term1(), indices1(), AD<T>(eye), eyeIndex);
  }
  const Shape& shapeTerm1Impl() const final { return this->term1().shape(); }
  const Shape& shapeTerm2Impl() const final { return this->term2().shape(); }
  const Shape& shapeImpl() const final { return resultShape_; }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Multiply>(*this);
  }

  std::string expressionImpl() const {
    std::ostringstream sout;
    sout << "multiply(" << this->term1().expression() << ", (";
    for (const auto& index : indices1()) {
      sout << index << " ";
    }
    sout << "), " << this->term2().expression() << ", (";
    for (const auto& index : indices2()) {
      sout << index << " ";
    }
    sout << "))";
    return sout.str();
  }

  AD<T> simplifyImpl() const final;

  Indices indices1_;
  Indices indices2_;
  Shape resultShape_;
};

// Multiply
template <typename T>
AD<T> Multiply<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  auto term1 = this->term1().simplify();
  auto term2 = this->term2().simplify();

  if (term1.template isType<Const>() && term2.template isType<Const>()) {
    return AD<T>(f(value(term1), value(term2)));
  }

  if ((term1.template isType<Const>() &&
       value(term1) == T::zeros(term1.shape())) ||
      (term2.template isType<Const>() &&
       value(term2) == T::zeros(term2.shape()))) {
    return AD<T>(T::zeros(this->shape()));
  }

  /*
  // 1 * Exression -> Expression
  if (term1.template isType<Const>() && Util::almostEqual(value(term1), 1)) {
    return term2;
  }
  */

  return multiply(term1, indices1(), term2, indices2());
}

template <typename T>
AD<T> multiply(const AD<T>& term1, const Indices& indices1, const AD<T>& term2,
               const Indices& indices2) {
  return Multiply<T>::makeAD(term1, indices1, term2, indices2);
}

template <typename T>
AD<T> operator*(const typename T::ValueType& scalar, const AD<T>& term) {
  Indices indices(term.shape().nDimensions());
  std::iota(indices.begin(), indices.end(), 0);

  return multiply(AD<T>(T::fill(term.shape(), scalar)), indices, term, indices);
}

template <typename T>
AD<T> operator*(const AD<T>& term, const typename T::ValueType& scalar) {
  return operator*(scalar, term);
}

template <typename T>
AD<T> operator/(const AD<T>& term, const typename T::ValueType& scalar) {
  return operator*(1.0 / scalar, term);
}

}  // namespace Alexandria

#endif  // AUTOMATIC_DIFFERENTIATION_AD_BINARY_H_
