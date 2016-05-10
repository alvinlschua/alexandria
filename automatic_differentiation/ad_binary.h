#ifndef AUTOMATIC_DIFFERENTIATION_AD_BINARY_H_
#define AUTOMATIC_DIFFERENTIATION_AD_BINARY_H_

#include <locale>
#include <memory>
#include <string>

#include "automatic_differentiation/ad.h"
#include "automatic_differentiation/ad_expression.h"
#include "automatic_differentiation/ad_var.h"
#include "automatic_differentiation/ad_const.h"
#include "util/clonable.h"
#include "util/util.h"

namespace AutomaticDifferentiation {

template <typename T>
class AD<T>::Binary : public Expression {
 public:
  using VarValues = AD<T>::VarValues;

  Binary(const AD<T>& ad1, const AD<T>& ad2) : ad1_(ad1), ad2_(ad2) {}
  Binary(const Binary&) = default;
  Binary& operator=(const Binary&) = default;

  virtual ~Binary() {}

  const AD<T>& adFirst() const { return ad1_; }
  const AD<T>& adSecond() const { return ad2_; }

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

  // Evaluate the expression.
  AD<T> evaluateAtImpl(const VarValues& varValues) const final;

  AD<T> ad1_;
  AD<T> ad2_;
};

template <typename T>
AD<T> AD<T>::Binary::differentiateImpl(const AD<T>& var) const {
  // TODO(alvin) Only reverse mode at the moment. Consider implementing forward
  // mode.
  auto result = dF1() * adFirst().differentiate(var) +
                dF2() * adSecond().differentiate(var);

  return result.simplify();
}

template <typename T>
AD<T> AD<T>::Binary::evaluateAtImpl(const VarValues& varValues) const {
  auto ad1 = (!adFirst().template isType<Const>())
                 ? adFirst().evaluateAt(varValues)
                 : adFirst();
  auto ad2 = (!adSecond().template isType<Const>())
                 ? adSecond().evaluateAt(varValues)
                 : adSecond();

  ad1 = ad1.simplify();
  ad2 = ad2.simplify();

  if (ad1.template isType<Const>() && ad2.template isType<Const>()) {
    return AD<T>(f(value(ad1), value(ad2)));
  }

  auto ptr = this->clone();
  dynamic_cast<Binary*>(ptr.get())->ad1_ = ad1;
  dynamic_cast<Binary*>(ptr.get())->ad2_ = ad2;

  return AD<T>(std::move(ptr)).simplify();
}

template <typename T>
class Plus : public AD<T>::Binary {
 public:
  Plus(const Plus&) = default;
  Plus& operator=(const Plus&) = default;
  virtual ~Plus() {}

  static AD<T> makeAD(const AD<T>& ad1, const AD<T>& ad2) {
    return AD<T>(Plus(ad1, ad2).clone());
  }

 private:
  Plus(const AD<T>& ad1, const AD<T>& ad2) : AD<T>::Binary(ad1, ad2) {}

  T f(const T& value1, const T& value2) const final { return value1 + value2; }
  AD<T> dF1() const final { return AD<T>(1.0); }
  AD<T> dF2() const final { return AD<T>(1.0); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Plus>(*this);
  }

  std::string expressionImpl() const {
    return "(" + this->adFirst().expression() + " + " +
           this->adSecond().expression() + ")";
  }

  AD<T> simplifyImpl() const final;
};

template <typename T>
class Minus : public AD<T>::Binary {
 public:
  Minus(const Minus&) = default;
  Minus& operator=(const Minus&) = default;
  virtual ~Minus() {}

  static AD<T> makeAD(const AD<T>& ad1, const AD<T>& ad2) {
    return AD<T>(Minus(ad1, ad2).clone());
  }

 private:
  Minus(const AD<T>& ad1, const AD<T>& ad2) : AD<T>::Binary(ad1, ad2) {}

  T f(const T& value1, const T& value2) const final { return value1 - value2; }
  AD<T> dF1() const final { return AD<T>(1.0); }
  AD<T> dF2() const final { return AD<T>(-1.0); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Minus>(*this);
  }

  std::string expressionImpl() const {
    return "(" + this->adFirst().expression() + " - " +
           this->adSecond().expression() + ")";
  }

  AD<T> simplifyImpl() const;
};

template <typename T>
class Times : public AD<T>::Binary {
 public:
  Times(const Times&) = default;
  Times& operator=(const Times&) = default;
  virtual ~Times() {}

  static AD<T> makeAD(const AD<T>& ad1, const AD<T>& ad2) {
    return AD<T>(Times(ad1, ad2).clone());
  }

 private:
  Times(const AD<T>& ad1, const AD<T>& ad2) : AD<T>::Binary(ad1, ad2) {}

  T f(const T& value1, const T& value2) const final { return value1 * value2; }
  AD<T> dF1() const final { return this->adSecond(); }
  AD<T> dF2() const final { return this->adFirst(); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Times>(*this);
  }

  std::string expressionImpl() const {
    return this->adFirst().expression() + " * " + this->adSecond().expression();
  }

  AD<T> simplifyImpl() const final;
};

template <typename T>
class Divide : public AD<T>::Binary {
 public:
  Divide(const Divide&) = default;
  Divide& operator=(const Divide&) = default;
  virtual ~Divide() {}

  static AD<T> makeAD(const AD<T>& ad1, const AD<T>& ad2) {
    return AD<T>(Divide(ad1, ad2).clone());
  }

 private:
  Divide(const AD<T>& ad1, const AD<T>& ad2) : AD<T>::Binary(ad1, ad2) {}

  T f(const T& value1, const T& value2) const { return value1 / value2; }
  AD<T> dF1() const final { return 1.0 / this->adSecond(); }
  AD<T> dF2() const final {
    return -this->adFirst() / (this->adSecond() * this->adSecond());
  }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Divide>(*this);
  }

  std::string expressionImpl() const {
    return this->adFirst().expression() + " / " + this->adSecond().expression();
  }

  AD<T> simplifyImpl() const final;
};

template <typename T>
class Pow : public AD<T>::Binary {
 public:
  Pow(const Pow&) = default;
  Pow& operator=(const Pow&) = default;
  virtual ~Pow() {}

  static AD<T> makeAD(const AD<T>& ad1, const AD<T>& ad2) {
    return AD<T>(Pow(ad1, ad2).clone());
  }

 private:
  Pow(const AD<T>& ad1, const AD<T>& ad2) : AD<T>::Binary(ad1, ad2) {}

  T f(const T& value1, const T& value2) const {
    return std::pow(value1, value2);
  }
  AD<T> dF1() const final {
    return this->adSecond() * pow(this->adFirst(), this->adSecond() - 1.0);
  }

  AD<T> dF2() const final {
    return pow(this->adFirst(), this->adSecond()) * log(this->adFirst());
  }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Pow>(*this);
  }

  std::string expressionImpl() const {
    return "pow(" + this->adFirst().expression() + " , " +
           this->adSecond().expression() + ")";
  }

  AD<T> simplifyImpl() const final;
};

// Plus
template <typename T>
AD<T> Plus<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  auto ad1 = this->adFirst().simplify();
  auto ad2 = this->adSecond().simplify();

  if (ad1.template isType<Const>() && ad2.template isType<Const>()) {
    return AD<T>(f(value(ad1), value(ad2)));
  }

  // Convert to Const + Expression if possible
  if (ad2.template isType<Const>()) {
    std::swap(ad1, ad2);
  }

  // 0 + Expression -> -Expression
  if (ad1.template isType<Const>() && Util::almostEqual(value(ad1), 0)) {
    return ad2;
  }

  // (c1 * Expression) + (c2 * Expression) = (c1 + c2) * Expression
  if (ad1.template isType<Times<T>>() && ad2.template isType<Times<T>>()) {
    auto& ad11 = ad1.template reference<Times<T>>().adFirst();
    auto& ad12 = ad1.template reference<Times<T>>().adSecond();
    auto& ad21 = ad2.template reference<Times<T>>().adFirst();
    auto& ad22 = ad2.template reference<Times<T>>().adSecond();

    if (ad11.template isType<Const>() && ad21.template isType<Const>() &&
        (ad12.expression() == ad22.expression())) {
      auto result = (ad11 + ad21) * ad12;
      return result.simplify();
    }
  }

  return ad1 + ad2;
}
template <typename T>
AD<T> operator+(const AD<T>& ad1, const AD<T>& ad2) {
  return Plus<T>::makeAD(ad1, ad2);
}

template <typename T>
AD<T> operator+(const T& value, const AD<T>& ad2) {
  return Plus<T>::makeAD(AD<T>(value), ad2);
}

template <typename T>
AD<T> operator+(const AD<T>& ad1, const T& value) {
  return Plus<T>::makeAD(ad1, AD<T>(value));
}

// Minus
template <typename T>
AD<T> Minus<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;
  using Times = Times<T>;

  auto ad1 = this->adFirst().simplify();
  auto ad2 = this->adSecond().simplify();

  if (ad1.template isType<Const>() && ad2.template isType<Const>()) {
    return AD<T>(f(value(ad1), value(ad2)));
  }

  // Convert to AD<T>::Const - Expression if possible
  if (ad2.template isType<Const>()) {
    std::swap(ad1, ad2);
  }

  // 0 - Expression -> -Expression
  if (ad1.template isType<Const>() && Util::almostEqual(value(ad1), 0)) {
    return -ad2;
  }

  // (c1 * Expression) + (c2 * Expression) = (c1 + c2) * Expression
  if (ad1.template isType<Times>() && ad2.template isType<Times>()) {
    auto& ad11 = ad1.template reference<Times>().adFirst();
    auto& ad12 = ad1.template reference<Times>().adSecond();
    auto& ad21 = ad2.template reference<Times>().adFirst();
    auto& ad22 = ad2.template reference<Times>().adSecond();

    if (ad11.template isType<Const>() && ad21.template isType<Const>() &&
        (ad12.expression() == ad22.expression())) {
      auto result = (ad11 + ad21) * ad12;
      return result.simplify();
    }
  }

  return ad1 - ad2;
}

template <typename T>
AD<T> operator-(const AD<T>& ad1, const AD<T>& ad2) {
  return Minus<T>::makeAD(ad1, ad2);
}

template <typename T>
AD<T> operator-(const T& value, const AD<T>& ad2) {
  return Minus<T>::makeAD(AD<T>(value), ad2);
}

template <typename T>
AD<T> operator-(const AD<T>& ad1, const T& value) {
  return Minus<T>::makeAD(ad1, AD<T>(value));
}

// Times
template <typename T>
AD<T> Times<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  auto ad1 = this->adFirst().simplify();
  auto ad2 = this->adSecond().simplify();

  if (ad1.template isType<Const>() && ad2.template isType<Const>()) {
    return AD<T>(f(value(ad1), value(ad2)));
  }

  // Convert to AD<T>::Const * Expression if possible
  if (ad2.template isType<Const>()) {
    std::swap(ad1, ad2);
  }

  // 0 * Exression -> 0
  if (ad1.template isType<Const>() && Util::almostEqual(value(ad1), 0)) {
    return AD<T>(0);
  }

  // 1 * Exression -> Expression
  if (ad1.template isType<Const>() && Util::almostEqual(value(ad1), 1)) {
    return ad2;
  }

  if (ad2.template isType<Times>()) {
    // AD<T>::Const * (AD<T>::Const * Expression) -> AD<T>::Const * expression
    auto ad2Ptr = ad2.template pointer<Times>();
    if (ad1.template isType<Const>() &&
        ad2Ptr->adFirst().template isType<Const>()) {
      auto result =
          AD<T>(f(value(ad1), value(ad2Ptr->adFirst()))) * ad2Ptr->adSecond();
      return result.simplify();
    }
  }

  return ad1 * ad2;
}

template <typename T>
AD<T> operator*(const AD<T>& ad1, const AD<T>& ad2) {
  return Times<T>::makeAD(ad1, ad2);
}

template <typename T>
AD<T> operator*(const T& value, const AD<T>& ad2) {
  return Times<T>::makeAD(AD<T>(value), ad2);
}

template <typename T>
AD<T> operator*(const AD<T>& ad1, const T& value) {
  return Times<T>::makeAD(ad1, AD<T>(value));
}

// Divide
template <typename T>
AD<T> Divide<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  auto ad1 = this->adFirst().simplify();
  auto ad2 = this->adSecond().simplify();

  if (ad1.template isType<Const>() && ad2.template isType<Const>()) {
    return AD<T>(f(value(ad1), value(ad2)));
  }

  // Expression / AD<T>::Const -> AD<T>::Const * Expression;
  if (ad2.template isType<Const>()) {
    return AD<T>(1.0 / value(ad2)) * ad1;
  }

  return ad1 / ad2;
}

template <typename T>
AD<T> operator/(const AD<T>& ad1, const AD<T>& ad2) {
  return Divide<T>::makeAD(ad1, ad2);
}

template <typename T>
AD<T> operator/(const T& value, const AD<T>& ad2) {
  return Divide<T>::makeAD(AD<T>(value), ad2);
}

template <typename T>
AD<T> operator/(const AD<T>& ad1, const T& value) {
  return Divide<T>::makeAD(ad1, AD<T>(value));
}

// Pow
template <typename T>
AD<T> Pow<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  auto ad1 = this->adFirst().simplify();
  auto ad2 = this->adSecond().simplify();

  if (ad1.template isType<Const>() && ad2.template isType<Const>()) {
    return AD<T>(f(value(ad1), value(ad2)));
  }

  // Expression / AD<T>::Const -> AD<T>::Const * Expression;
  if (ad2.template isType<Const>()) {
    if (Util::almostEqual(value(ad2), 1)) {
      return ad1;
    }
    if (Util::almostEqual(value(ad2), 0)) {
      return AD<T>(1);
    }
  }

  return pow(ad1, ad2);
}

template <typename T>
AD<T> pow(const AD<T>& ad1, const AD<T>& ad2) {
  return Pow<T>::makeAD(ad1, ad2);
}
template <typename T>
AD<T> pow(const AD<T>& ad, const T& value) {
  return Pow<T>::makeAD(ad, AD<T>(value));
}
template <typename T>
AD<T> pow(const T& value, const AD<T>& ad) {
  return Pow<T>::makeAD(AD<T>(value), ad);
}

}  // namespace AutomaticDifferentiation

#endif  // AUTOMATIC_DIFFERENTIATION_AD_BINARY_H_
