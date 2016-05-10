#ifndef AUTOMATIC_DIFFERENTIATION_AD_CONST_H_
#define AUTOMATIC_DIFFERENTIATION_AD_CONST_H_

#include <string>

#include "automatic_differentiation/ad_expression.h"

namespace AutomaticDifferentiation {

class AD::Const : public Expression {
 public:
  using VarValues = AD::VarValues;

  static std::unique_ptr<Expression> make(double value) {
    return Const(value).clone();
  }

  Const(const Const&) = default;
  Const& operator=(const Const&) = default;
  virtual ~Const() {}

  double value() const { return value_; }

 private:
  explicit Const(double value) : value_(value) {}

  AD differentiateImpl(const AD& /*var*/) const final { return AD(0); }

  AD simplifyImpl() const final { return AD(value()); }

  AD evaluateAtImpl(const VarValues& /*varValues*/) const final {
    return AD(value());
  }

  std::string expressionImpl() const final { return std::to_string(value()); }

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Const>(*this);
  }

  double value_;
};

}  // namespace AutomaticDifferentiation

#endif  // AUTOMATIC_DIFFERENTIATION_AD_CONST_H_
