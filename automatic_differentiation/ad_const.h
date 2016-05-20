#ifndef AUTOMATIC_DIFFERENTIATION_AD_CONST_H_
#define AUTOMATIC_DIFFERENTIATION_AD_CONST_H_

#include <string>

#include "automatic_differentiation/ad_expression.h"

namespace Alexandria {

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
  explicit Const(const T& value) : value_(value) {}

  AD<T> differentiateImpl(const AD<T>& /*var*/) const final { return AD<T>(0); }

  bool dependsOnImpl(const AD<T>& /*var*/) const final { return false; }

  AD<T> simplifyImpl() const final { return AD<T>(value()); }

  AD<T> evaluateAtImpl(const VarValues& /*varValues*/) const final {
    return AD<T>(value());
  }

  std::string expressionImpl() const final;

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Const>(*this);
  }

  T value_;
};

template <typename T>
std::string AD<T>::Const::expressionImpl() const {
  std::ostringstream sout;
  sout << value();
  return sout.str();
}

}  // namespace Alexandria 

#endif  // AUTOMATIC_DIFFERENTIATION_AD_CONST_H_
