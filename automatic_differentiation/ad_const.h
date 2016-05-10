#ifndef AUTOMATIC_DIFFERENTIATION_AD_CONST_H_
#define AUTOMATIC_DIFFERENTIATION_AD_CONST_H_

#include <string>

#include "automatic_differentiation/ad_expression.h"

namespace AutomaticDifferentiation {

template <typename T>
class AD<T>::Const : public Expression {
 public:
  using VarValues = AD::VarValues;

  static std::unique_ptr<Expression> make(const T& value) {
    return Const(value).clone();
  }

  Const(const Const&) = default;
  Const& operator=(const Const&) = default;
  virtual ~Const() {}

  const T& value() const { return value_; }

 private:
  explicit Const(double value) : value_(value) {}

  AD<T> differentiateImpl(const AD<T>& /*var*/) const final { return AD<T>(0); }

  AD<T> simplifyImpl() const final { return AD<T>(value()); }

  AD<T> evaluateAtImpl(const VarValues& /*varValues*/) const final {
    return AD<T>(value());
  }

  std::string expressionImpl() const final { return std::to_string(value()); }

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Const>(*this);
  }

  T value_;
};

}  // namespace AutomaticDifferentiation

#endif  // AUTOMATIC_DIFFERENTIATION_AD_CONST_H_
