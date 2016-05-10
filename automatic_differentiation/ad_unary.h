#ifndef AUTOMATIC_DIFFERENTIATION_AD_UNARY_H_
#define AUTOMATIC_DIFFERENTIATION_AD_UNARY_H_

#include <locale>
#include <memory>
#include <string>

#include "automatic_differentiation/ad.h"
#include "automatic_differentiation/ad_expression.h"
#include "automatic_differentiation/ad_var.h"
#include "automatic_differentiation/ad_const.h"
#include "util/clonable.h"

namespace AutomaticDifferentiation {

class AD::Unary : public Expression {
 public:
  using VarValues = AD::VarValues;

  explicit Unary(const AD& ad) : ad_(ad) {}
  Unary(const Unary&) = default;
  Unary& operator=(const Unary&) = default;

  virtual ~Unary() {}

 protected:
  AD ad_;

 private:
  // Value evaluation of the function itself.
  virtual double f(double value) const = 0;

  // Differential of the function with respect to the expression.
  virtual AD dF() const = 0;

  // Differentiate with respect to var.
  AD differentiateImpl(const AD& var) const final;

  // Evaluate the expression.
  AD evaluateAtImpl(const VarValues& varValues) const final;

  // Simplify the expression.
  AD simplifyImpl() const override;
};

class UnaryMinus : public AD::Unary {
 public:
  using VarValues = AD::VarValues;

  UnaryMinus(const UnaryMinus&) = default;
  UnaryMinus& operator=(const UnaryMinus&) = default;
  virtual ~UnaryMinus() {}

  static AD makeAD(const AD& ad) { return AD(UnaryMinus(ad).clone()); }

 private:
  explicit UnaryMinus(const AD& ad) : Unary(ad) {}

  double f(double value) const final;
  AD dF() const final;

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<UnaryMinus>(*this);
  }

  std::string expressionImpl() const {
    return ad_.isType<const AD::Const>() || ad_.isType<const AD::Var>()
               ? "-" + ad_.expression()
               : "-(" + ad_.expression() + ")";
  }

  AD simplifyImpl() const final;
};

AD operator-(const AD& ad);

class Sin : public AD::Unary {
 public:
  using VarValues = AD::VarValues;

  virtual ~Sin() {}
  Sin(const Sin&) = default;
  Sin& operator=(const Sin&) = default;

  static AD makeAD(const AD& ad) { return AD(Sin(ad).clone()); }

 private:
  explicit Sin(const AD& ad) : AD::Unary(ad) {}

  double f(double value) const final;
  AD dF() const final;

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Sin>(*this);
  }

  std::string expressionImpl() const { return "sin(" + ad_.expression() + ")"; }
};

AD sin(const AD& ad);

class Cos : public AD::Unary {
 public:
  using VarValues = AD::VarValues;

  virtual ~Cos() {}
  Cos(const Cos&) = default;
  Cos& operator=(const Cos&) = default;

  static AD makeAD(const AD& ad) { return AD(Cos(ad).clone()); }

 private:
  explicit Cos(const AD& ad) : Unary(ad) {}

  double f(double value) const final;
  AD dF() const final;

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Cos>(*this);
  }

  std::string expressionImpl() const { return "cos(" + ad_.expression() + ")"; }
};

AD cos(const AD& ad);

class Exp : public AD::Unary {
 public:
  using VarValues = AD::VarValues;

  virtual ~Exp() {}
  Exp(const Exp&) = default;
  Exp& operator=(const Exp&) = default;

  static AD makeAD(const AD& ad) { return AD(Exp(ad).clone()); }

 private:
  explicit Exp(const AD& ad) : Unary(ad) {}

  double f(double value) const final;
  AD dF() const final;

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Exp>(*this);
  }

  std::string expressionImpl() const { return "exp(" + ad_.expression() + ")"; }
};

AD exp(const AD& ad);

class Log : public AD::Unary {
 public:
  using VarValues = AD::VarValues;

  virtual ~Log() {}
  Log(const Log&) = default;
  Log& operator=(const Log&) = default;

  static AD makeAD(const AD& ad) { return AD(Log(ad).clone()); }

 private:
  explicit Log(const AD& ad) : Unary(ad) {}

  double f(double value) const final;
  AD dF() const final;

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Log>(*this);
  }

  std::string expressionImpl() const { return "exp(" + ad_.expression() + ")"; }
};

AD log(const AD& ad);

}  // namespace AutomaticDifferentiation

#endif  // AUTOMATIC_DIFFERENTIATION_AD_UNARY_H_
