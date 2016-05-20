#ifndef AUTOMATIC_DIFFERENTIATION_AD_PARAM_H_
#define AUTOMATIC_DIFFERENTIATION_AD_PARAM_H_

#include <locale>
#include <memory>
#include <string>

#include "automatic_differentiation/ad.h"
#include "automatic_differentiation/ad_const.h"
#include "util/clonable.h"

namespace Alexandria {

template <typename T>
class AD<T>::Param : public Expression {
 public:
  using VarValues = AD<T>::VarValues;

  static std::unique_ptr<Expression> make(const std::string& identifier,
                                          const T& value) {
    return Param(identifier, value).clone();
  }
  Param(const Param&) = default;
  Param& operator=(const Param&) = default;
  virtual ~Param() {}

  const std::string& identifier() const { return identifier_; }
  const T& value() const { return *value_; }
  T& value() { return *value_; }

 private:
  Param(const std::string& identifier, const T& value)
      : identifier_(identifier), value_(std::make_shared<T>(value)) {}

  AD<T> differentiateImpl(const AD<T>& var) const final {
    return AD<T>(identifier() == Alexandria::identifier(var) ? 1 : 0);
  }

  bool dependsOnImpl(const AD<T>& var) const final {
    return identifier() == Alexandria::identifier(var) ? true : false;
  }

  AD<T> evaluateAtImpl(const VarValues& /*varValues*/) const final {
    return AD<T>(value());
  }

  AD<T> simplifyImpl() const final { return AD(this->clone()); }

  std::string expressionImpl() const final { return identifier(); }

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Param>(*this);
  }

  std::string identifier_;
  std::shared_ptr<T> value_;
};

}  // namespace Alexandria

#endif  // AUTOMATIC_DIFFERENTIATION_AD_PARAM_H_
