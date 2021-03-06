#ifndef AUTOMATIC_DIFFERENTIATION_AD_BINARY_H_
#define AUTOMATIC_DIFFERENTIATION_AD_BINARY_H_

#include <locale>
#include <memory>
#include <string>

#include "automatic_differentiation/ad.h"
#include "automatic_differentiation/ad_const.h"
#include "automatic_differentiation/ad_expression.h"
#include "automatic_differentiation/ad_var.h"
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

 private:
  // TODO(alvin) Considering adding fMove

  // Evaluate the unary function.
  virtual T f(const T& value1, const T& value2) const = 0;

  // Differential of the function with respect to the first expression.
  virtual AD<T> dF1() const = 0;

  // Differential of the function with respect to the second expression.
  virtual AD<T> dF2() const = 0;

  // Differentiate with respect to var.
  AD<T> differentiateImpl(const AD<T>& var) const final;

  // Does the function depend on the variable?
  bool dependsOnImpl(const AD<T>& var) const final {
    return term1().dependsOn(var) || term2().dependsOn(var);;
  }

  // Evaluate the expression.
  AD<T> evaluateAtImpl(const VarValues& varValues) const final;

  AD<T> term1_;
  AD<T> term2_;
};

template <typename T>
AD<T> AD<T>::Binary::differentiateImpl(const AD<T>& var) const {
  // TODO(alvin) Only reverse mode at the moment. Consider implementing forward
  // mode.
  auto result =
      dF1() * term1().differentiate(var) + dF2() * term2().differentiate(var);

  return result.simplify();
}

template <typename T>
AD<T> AD<T>::Binary::evaluateAtImpl(const VarValues& varValues) const {
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
  Plus(const AD<T>& term1, const AD<T>& term2) : AD<T>::Binary(term1, term2) {}

  T f(const T& value1, const T& value2) const final { return value1 + value2; }
  AD<T> dF1() const final { return AD<T>(1.0); }
  AD<T> dF2() const final { return AD<T>(1.0); }

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
class Minus : public AD<T>::Binary {
 public:
  Minus(const Minus&) = default;
  Minus& operator=(const Minus&) = default;
  virtual ~Minus() {}

  static AD<T> makeAD(const AD<T>& term1, const AD<T>& term2) {
    return AD<T>(Minus(term1, term2).clone());
  }

 private:
  Minus(const AD<T>& term1, const AD<T>& term2) : AD<T>::Binary(term1, term2) {}

  T f(const T& value1, const T& value2) const final { return value1 - value2; }
  AD<T> dF1() const final { return AD<T>(1.0); }
  AD<T> dF2() const final { return AD<T>(-1.0); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Minus>(*this);
  }

  std::string expressionImpl() const {
    return "(" + this->term1().expression() + " - " +
           this->term2().expression() + ")";
  }

  AD<T> simplifyImpl() const;
};

template <typename T>
class Times : public AD<T>::Binary {
 public:
  Times(const Times&) = default;
  Times& operator=(const Times&) = default;
  virtual ~Times() {}

  static AD<T> makeAD(const AD<T>& term1, const AD<T>& term2) {
    return AD<T>(Times(term1, term2).clone());
  }

 private:
  Times(const AD<T>& term1, const AD<T>& term2) : AD<T>::Binary(term1, term2) {}

  T f(const T& value1, const T& value2) const final { return value1 * value2; }
  AD<T> dF1() const final { return this->term2(); }
  AD<T> dF2() const final { return this->term1(); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Times>(*this);
  }

  std::string expressionImpl() const {
    return this->term1().expression() + " * " + this->term2().expression();
  }

  AD<T> simplifyImpl() const final;
};

template <typename T>
class Divide : public AD<T>::Binary {
 public:
  Divide(const Divide&) = default;
  Divide& operator=(const Divide&) = default;
  virtual ~Divide() {}

  static AD<T> makeAD(const AD<T>& term1, const AD<T>& term2) {
    return AD<T>(Divide(term1, term2).clone());
  }

 private:
  Divide(const AD<T>& term1, const AD<T>& term2)
      : AD<T>::Binary(term1, term2) {}

  T f(const T& value1, const T& value2) const { return value1 / value2; }
  AD<T> dF1() const final { return 1.0 / this->term2(); }
  AD<T> dF2() const final {
    return -this->term1() / (this->term2() * this->term2());
  }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Divide>(*this);
  }

  std::string expressionImpl() const {
    return this->term1().expression() + " / " + this->term2().expression();
  }

  AD<T> simplifyImpl() const final;
};

template <typename T>
class Pow : public AD<T>::Binary {
 public:
  Pow(const Pow&) = default;
  Pow& operator=(const Pow&) = default;
  virtual ~Pow() {}

  static AD<T> makeAD(const AD<T>& term1, const AD<T>& term2) {
    return AD<T>(Pow(term1, term2).clone());
  }

 private:
  Pow(const AD<T>& term1, const AD<T>& term2) : AD<T>::Binary(term1, term2) {}

  T f(const T& value1, const T& value2) const {
    return std::pow(value1, value2);
  }
  AD<T> dF1() const final {
    return this->term2() * pow(this->term1(), this->term2() - 1.0);
  }

  AD<T> dF2() const final {
    return pow(this->term1(), this->term2()) * log(this->term1());
  }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Pow>(*this);
  }

  std::string expressionImpl() const {
    return "pow(" + this->term1().expression() + " , " +
           this->term2().expression() + ")";
  }

  AD<T> simplifyImpl() const final;
};

// Plus
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

  // 0 + Expression -> -Expression
  if (term1.template isType<Const>() && almostEqual(value(term1), 0)) {
    return term2;
  }

  // (c1 * Expression) + (c2 * Expression) = (c1 + c2) * Expression
  if (term1.template isType<Times<T>>() && term2.template isType<Times<T>>()) {
    auto& term11 = term1.template reference<Times<T>>().term1();
    auto& term12 = term1.template reference<Times<T>>().term2();
    auto& term21 = term2.template reference<Times<T>>().term1();
    auto& term22 = term2.template reference<Times<T>>().term2();

    if (term11.template isType<Const>() && term21.template isType<Const>() &&
        (term12.expression() == term22.expression())) {
      auto result = (term11 + term21) * term12;
      return result.simplify();
    }
  }

  return term1 + term2;
}
template <typename T>
AD<T> operator+(const AD<T>& term1, const AD<T>& term2) {
  return Plus<T>::makeAD(term1, term2);
}

template <typename T>
AD<T> operator+(const T& value, const AD<T>& term2) {
  return Plus<T>::makeAD(AD<T>(value), term2);
}

template <typename T>
AD<T> operator+(const AD<T>& term1, const T& value) {
  return Plus<T>::makeAD(term1, AD<T>(value));
}

// Minus
template <typename T>
AD<T> Minus<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;
  using Times = Times<T>;

  auto term1 = this->term1().simplify();
  auto term2 = this->term2().simplify();

  if (term1.template isType<Const>() && term2.template isType<Const>()) {
    return AD<T>(f(value(term1), value(term2)));
  }

  // 0 - Expression -> -Expression
  if (term1.template isType<Const>() && almostEqual(value(term1), 0)) {
    return -term2;
  }

  // Expression - 0 -> Expression
  if (term2.template isType<Const>() && almostEqual(value(term2), 0)) {
    return term1;
  }

  // (c1 * Expression) - (c2 * Expression) = (c1 + c2) * Expression
  if (term1.template isType<Times>() && term2.template isType<Times>()) {
    auto& term11 = term1.template reference<Times>().term1();
    auto& term12 = term1.template reference<Times>().term2();
    auto& term21 = term2.template reference<Times>().term1();
    auto& term22 = term2.template reference<Times>().term2();

    if (term11.template isType<Const>() && term21.template isType<Const>() &&
        (term12.expression() == term22.expression())) {
      auto result = (term11 - term21) * term12;
      return result.simplify();
    }
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

// Times
template <typename T>
AD<T> Times<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  auto term1 = this->term1().simplify();
  auto term2 = this->term2().simplify();

  if (term1.template isType<Const>() && term2.template isType<Const>()) {
    return AD<T>(f(value(term1), value(term2)));
  }

  // Convert to AD<T>::Const * Expression if possible
  if (term2.template isType<Const>()) {
    std::swap(term1, term2);
  }

  // 0 * Exression -> 0
  if (term1.template isType<Const>() && almostEqual(value(term1), 0)) {
    return AD<T>(0);
  }

  // 1 * Exression -> Expression
  if (term1.template isType<Const>() && almostEqual(value(term1), 1)) {
    return term2;
  }

  if (term2.template isType<Times>()) {
    // AD<T>::Const * (AD<T>::Const * Expression) -> AD<T>::Const * expression
    auto term2Ptr = term2.template pointer<Times>();
    if (term1.template isType<Const>() &&
        term2Ptr->term1().template isType<Const>()) {
      auto result =
          AD<T>(f(value(term1), value(term2Ptr->term1()))) * term2Ptr->term2();
      return result.simplify();
    }
  }

  return term1 * term2;
}

template <typename T>
AD<T> operator*(const AD<T>& term1, const AD<T>& term2) {
  return Times<T>::makeAD(term1, term2);
}

template <typename T>
AD<T> operator*(const T& value, const AD<T>& term2) {
  return Times<T>::makeAD(AD<T>(value), term2);
}

template <typename T>
AD<T> operator*(const AD<T>& term1, const T& value) {
  return Times<T>::makeAD(term1, AD<T>(value));
}

// Divide
template <typename T>
AD<T> Divide<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  auto term1 = this->term1().simplify();
  auto term2 = this->term2().simplify();

  if (term1.template isType<Const>() && term2.template isType<Const>()) {
    return AD<T>(f(value(term1), value(term2)));
  }

  // Expression / AD<T>::Const -> AD<T>::Const * Expression;
  if (term2.template isType<Const>()) {
    return AD<T>(1.0 / value(term2)) * term1;
  }

  return term1 / term2;
}

template <typename T>
AD<T> operator/(const AD<T>& term1, const AD<T>& term2) {
  return Divide<T>::makeAD(term1, term2);
}

template <typename T>
AD<T> operator/(const T& value, const AD<T>& term2) {
  return Divide<T>::makeAD(AD<T>(value), term2);
}

template <typename T>
AD<T> operator/(const AD<T>& term1, const T& value) {
  return Divide<T>::makeAD(term1, AD<T>(value));
}

// Pow
template <typename T>
AD<T> Pow<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  auto term1 = this->term1().simplify();
  auto term2 = this->term2().simplify();

  if (term1.template isType<Const>() && term2.template isType<Const>()) {
    return AD<T>(f(value(term1), value(term2)));
  }

  // Expression / AD<T>::Const -> AD<T>::Const * Expression;
  if (term2.template isType<Const>()) {
    if (almostEqual(value(term2), 1)) {
      return term1;
    }
    if (almostEqual(value(term2), 0)) {
      return AD<T>(1);
    }
  }

  return pow(term1, term2);
}

template <typename T>
AD<T> pow(const AD<T>& term1, const AD<T>& term2) {
  return Pow<T>::makeAD(term1, term2);
}
template <typename T>
AD<T> pow(const AD<T>& ad, const T& value) {
  return Pow<T>::makeAD(ad, AD<T>(value));
}
template <typename T>
AD<T> pow(const T& value, const AD<T>& ad) {
  return Pow<T>::makeAD(AD<T>(value), ad);
}

}  // namespace Alexandria

#endif  // AUTOMATIC_DIFFERENTIATION_AD_BINARY_H_
