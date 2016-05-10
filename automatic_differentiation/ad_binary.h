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

namespace AutomaticDifferentiation {

class AD::Binary : public Expression {
 public:
  using VarValues = AD::VarValues;

  Binary(const AD& ad1, const AD& ad2) : ad1_(ad1), ad2_(ad2) {}
  Binary(const Binary&) = default;
  Binary& operator=(const Binary&) = default;

  virtual ~Binary() {}

  const AD& adFirst() const { return ad1_; }
  const AD& adSecond() const { return ad2_; }

 private:
  // Evaluate the unary function.
  virtual double f(double value1, double value2) const = 0;

  // Differential of the function with respect to the first expression.
  virtual AD dF1() const = 0;

  // Differential of the function with respect to the second expression.
  virtual AD dF2() const = 0;

  // Differentiate with respect to var.
  AD differentiateImpl(const AD& var) const final;

  // Evaluate the expression.
  AD evaluateAtImpl(const VarValues& varValues) const final;

  AD ad1_;
  AD ad2_;
};

class Plus : public AD::Binary {
 public:
  using VarValues = AD::VarValues;

  Plus(const Plus&) = default;
  Plus& operator=(const Plus&) = default;
  virtual ~Plus() {}

  static AD makeAD(const AD& ad1, const AD& ad2) {
    return AD(Plus(ad1, ad2).clone());
  }

 private:
  Plus(const AD& ad1, const AD& ad2) : Binary(ad1, ad2) {}

  double f(double value1, double value2) const final;
  AD dF1() const final;
  AD dF2() const final;

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Plus>(*this);
  }

  std::string expressionImpl() const {
    return "(" + adFirst().expression() + " + " + adSecond().expression() + ")";
  }

  AD simplifyImpl() const final;
};

AD operator+(const AD& ad1, const AD& ad2);

AD operator+(double value, const AD& ad2);

AD operator+(const AD& ad1, double value);

class Minus : public AD::Binary {
 public:
  using VarValues = AD::VarValues;

  Minus(const Minus&) = default;
  Minus& operator=(const Minus&) = default;
  virtual ~Minus() {}

  static AD makeAD(const AD& ad1, const AD& ad2) {
    return AD(Minus(ad1, ad2).clone());
  }

 private:
  Minus(const AD& ad1, const AD& ad2) : Binary(ad1, ad2) {}

  double f(double value1, double value2) const final;
  AD dF1() const final;
  AD dF2() const final;

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Minus>(*this);
  }

  std::string expressionImpl() const {
    return "(" + adFirst().expression() + " - " + adSecond().expression() + ")";
  }

  AD simplifyImpl() const;
};

AD operator-(const AD& ad1, const AD& ad2);

AD operator-(double value, const AD& ad2);

AD operator-(const AD& ad1, double value);

class Times : public AD::Binary {
 public:
  using VarValues = AD::VarValues;

  Times(const Times&) = default;
  Times& operator=(const Times&) = default;
  virtual ~Times() {}

  static AD makeAD(const AD& ad1, const AD& ad2) {
    return AD(Times(ad1, ad2).clone());
  }

 private:
  Times(const AD& ad1, const AD& ad2) : Binary(ad1, ad2) {}

  double f(double value1, double value2) const final;
  AD dF1() const final;
  AD dF2() const final;

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Times>(*this);
  }

  std::string expressionImpl() const {
    return adFirst().expression() + " * " + adSecond().expression();
  }

  AD simplifyImpl() const final;
};

AD operator*(const AD& ad1, const AD& ad2);

AD operator*(double value, const AD& ad2);

AD operator*(const AD& ad1, double value);

class Divide : public AD::Binary {
 public:
  using VarValues = AD::VarValues;

  Divide(const Divide&) = default;
  Divide& operator=(const Divide&) = default;
  virtual ~Divide() {}

  static AD makeAD(const AD& ad1, const AD& ad2) {
    return AD(Divide(ad1, ad2).clone());
  }

 private:
  Divide(const AD& ad1, const AD& ad2) : Binary(ad1, ad2) {}

  double f(double value1, double value2) const;
  AD dF1() const final;
  AD dF2() const final;

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Divide>(*this);
  }

  std::string expressionImpl() const {
    return adFirst().expression() + " / " + adSecond().expression();
  }

  AD simplifyImpl() const final;
};

AD operator/(const AD& ad1, const AD& ad2);

AD operator/(double value, const AD& ad2);

AD operator/(const AD& ad1, double value);

class Pow : public AD::Binary {
 public:
  using VarValues = AD::VarValues;

  Pow(const Pow&) = default;
  Pow& operator=(const Pow&) = default;
  virtual ~Pow() {}

  static AD makeAD(const AD& ad1, const AD& ad2) {
    return AD(Pow(ad1, ad2).clone());
  }

 private:
  Pow(const AD& ad1, const AD& ad2) : Binary(ad1, ad2) {}

  double f(double value1, double value2) const;
  AD dF1() const final;
  AD dF2() const final;

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Pow>(*this);
  }

  std::string expressionImpl() const {
    return "pow(" + adFirst().expression() + " , " + adSecond().expression() +
           ")";
  }

  AD simplifyImpl() const final;
};

AD pow(const AD& ad1, const AD& ad2);
AD pow(const AD& ad, double value);
AD pow(double value, const AD& ad);

}  // namespace AutomaticDifferentiation

#endif  // AUTOMATIC_DIFFERENTIATION_AD_BINARY_H_
