#ifndef AUTOMATIC_DIFFERENTIATION_AD_UNARY_H_
#define AUTOMATIC_DIFFERENTIATION_AD_UNARY_H_

#include <memory>
#include <string>

#include "automatic_differentiation/ad.h"
#include "automatic_differentiation/ad_expression.h"
#include "automatic_differentiation/ad_var.h"
#include "automatic_differentiation/ad_const.h"
#include "util/clonable.h"

namespace AutomaticDifferentiation {

template <typename T>
class AD<T>::Unary : public Expression {
 public:
  using VarValues = typename AD<T>::VarValues;

  explicit Unary(const AD<T>& ad) : ad_(ad) {}
  Unary(const Unary&) = default;
  Unary& operator=(const Unary&) = default;

  virtual ~Unary() {}

 protected:
  AD<T> ad_;

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
};

template <typename T>
AD<T> AD<T>::Unary::differentiateImpl(const AD<T>& var) const {
  // TODO(alvin) Only reverse mode at the moment. Consider implementing forward
  // mode.
  auto result = dF() * ad_.differentiate(var);
  return result.simplify();
}

template <typename T>
AD<T> AD<T>::Unary::evaluateAtImpl(const VarValues& varValues) const {
  using Const = typename AD<T>::Const;
  auto ad = !ad_.template isType<Const>() ? ad_.evaluateAt(varValues) : ad_;
  ad = ad.simplify();

  if (ad.template isType<Const>()) {
    return AD<T>(f(ad.template reference<Const>().value()));
  }

  auto ptr = this->clone();
  dynamic_cast<Unary*>(ptr.get())->ad_ = ad;

  return AD<T>(std::move(ptr));
}

template <typename T>
AD<T> AD<T>::Unary::simplifyImpl() const {
  return ad_.template isType<typename AD<T>::Const>() ? AD<T>(f(value(ad_)))
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
  AD<T> dF() const final { return AD<T>(-1); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<UnaryMinus>(*this);
  }

  std::string expressionImpl() const {
    return this->ad_.template isType<typename AD<T>::Const>() ||
                   this->ad_.template isType<typename AD<T>::Var>()
               ? "-" + this->ad_.expression()
               : "-(" + this->ad_.expression() + ")";
  }

  AD<T> simplifyImpl() const final;
};

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
  AD<T> dF() const final { return cos(this->ad_); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Sin>(*this);
  }

  std::string expressionImpl() const {
    return "sin(" + this->ad_.expression() + ")";
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
  AD<T> dF() const final { return -sin(this->ad_); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Cos>(*this);
  }

  std::string expressionImpl() const {
    return "cos(" + this->ad_.expression() + ")";
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
  AD<T> dF() const final { return exp(this->ad_); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Exp>(*this);
  }

  std::string expressionImpl() const {
    return "exp(" + this->ad_.expression() + ")";
  }
};

template <typename T>
class Log : public AD<T>::Unary {
 public:
  virtual ~Log() {}
  Log(const Log&) = default;
  Log& operator=(const Log&) = default;

  static AD<T> makeAD(const AD<T>& ad) { return AD<T>(Log(ad).clone()); }

 private:
  explicit Log(const AD<T>& ad) : AD<T>::Unary(ad) {}

  T f(const T& value) const final { return std::log(value); }
  AD<T> dF() const final { return 1.0 / this->ad_; }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Log>(*this);
  }

  std::string expressionImpl() const {
    return "exp(" + this->ad_.expression() + ")";
  }
};

// Unary Minus
template <typename T>
AD<T> UnaryMinus<T>::simplifyImpl() const {
  AD<T> result;
  if (this->ad_.template isType<typename AD<T>::Const>()) {
    result = AD<T>(f(value(this->ad_)));
  } else if (this->ad_.template isType<UnaryMinus>()) {
    result = this->ad_.template reference<UnaryMinus>().ad_;
  } else {
    result = AD<T>(this->clone());
  }
  return result;
}

template <typename T>
AD<T> operator-(const AD<T>& ad) {
  return UnaryMinus<T>::makeAD(ad);
}

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

}  // namespace AutomaticDifferentiation

#endif  // AUTOMATIC_DIFFERENTIATION_AD_UNARY_H_
