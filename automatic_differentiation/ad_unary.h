#ifndef AUTOMATIC_DIFFERENTIATION_AD_UNARY_H_
#define AUTOMATIC_DIFFERENTIATION_AD_UNARY_H_

#include <memory>
#include <string>

#include "automatic_differentiation/ad.h"
#include "automatic_differentiation/ad_const.h"
#include "automatic_differentiation/ad_expression.h"
#include "automatic_differentiation/ad_var.h"
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

 private:
  // Value evaluation of the function itself.
  virtual T f(const T& value) const = 0;

  // Differential of the function with respect to the expression.
  virtual AD<T> dF() const = 0;

  // Differentiate with respect to var.
  AD<T> differentiateImpl(const AD<T>& var) const final;

  // Does the function depend on the variable?
  bool dependsOnImpl(const AD<T>& var) const final {
    return term().dependsOn(var);
  }

  // Evaluate the expression.
  AD<T> evaluateAtImpl(const VarValues& varValues) const final;

  // Simplify the expression.
  AD<T> simplifyImpl() const override;

  AD<T> term_;
};

template <typename T>
AD<T> AD<T>::Unary::differentiateImpl(const AD<T>& var) const {
  // TODO(alvin) Only reverse mode at the moment. Consider implementing forward
  // mode.
  auto result = dF() * term().differentiate(var);
  return result.simplify();
}

template <typename T>
AD<T> AD<T>::Unary::evaluateAtImpl(const VarValues& varValues) const {
  auto term = this->term().template isType<Const>()
                  ? this->term()
                  : this->term().evaluateAt(varValues);
  term = term.simplify();

  if (term.template isType<Const>()) {
    return AD<T>(f(value(term)));
  }

  auto ptr = this->clone();
  dynamic_cast<Unary*>(ptr.get())->term_ = term;

  return AD<T>(std::move(ptr)).simplify();
}

template <typename T>
AD<T> AD<T>::Unary::simplifyImpl() const {
  using Const = typename AD<T>::Const;
  return term().template isType<Const>() ? AD<T>(f(value(term())))
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
    using Const = typename AD<T>::Const;
    using Var = typename AD<T>::Var;

    return this->term().template isType<Const>() ||
                   this->term().template isType<Var>()
               ? "-" + this->term().expression()
               : "-(" + this->term().expression() + ")";
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
  AD<T> dF() const final { return 1.0 / this->term(); }

  std::unique_ptr<typename AD<T>::Expression> cloneImpl() const {
    return std::make_unique<Log>(*this);
  }

  std::string expressionImpl() const {
    return "exp(" + this->term().expression() + ")";
  }
};

// Unary Minus
template <typename T>
AD<T> UnaryMinus<T>::simplifyImpl() const {
  using Const = typename AD<T>::Const;

  if (this->term().template isType<Const>()) {
    return AD<T>(f(value(this->term())));
  } else if (this->term().template isType<UnaryMinus>()) {
    return this->term().template reference<UnaryMinus>().term();
  }
  return AD<T>(this->clone());
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

}  // namespace Alexandria

#endif  // AUTOMATIC_DIFFERENTIATION_AD_UNARY_H_
