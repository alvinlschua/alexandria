#ifndef AUTOMATIC_DIFFERENTIATION_AD_VAR_H_
#define AUTOMATIC_DIFFERENTIATION_AD_VAR_H_

#include <locale>
#include <memory>
#include <string>

#include "automatic_differentiation/ad.h"
#include "automatic_differentiation/ad_const.h"
#include "util/clonable.h"

namespace AutomaticDifferentiation {

template <typename T>
class AD<T>::Var : public Expression {
 public:
  using VarValues = AD<T>::VarValues;

  static std::unique_ptr<Expression> make(const std::string& identifier) {
    return Var(identifier).clone();
  }
  Var(const Var&) = default;
  Var& operator=(const Var&) = default;

  virtual ~Var() {}

  const std::string& identifier() const { return identifier_; }

 private:
  explicit Var(const std::string& identifier) : identifier_(identifier) {}

  AD<T> differentiateImpl(const AD<T>& var) const final {
    return AD<T>(
        identifier() == AutomaticDifferentiation::identifier<T>(var) ? 1 : 0);
  }

  AD<T> evaluateAtImpl(const VarValues& varValues) const final {
    for (const auto& varValue : varValues) {
      if (identifier() ==
          AutomaticDifferentiation::identifier<T>(varValue.first)) {
        return AD<T>(varValue.second);
      }
    }
    return AD<T>(identifier());
  }

  AD<T> simplifyImpl() const final { return AD<T>(identifier()); }

  std::string expressionImpl() const final { return identifier(); }

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Var>(*this);
  }

  std::string identifier_;
};

}  // namespace AutomaticDifferentiation

#endif  // AUTOMATIC_DIFFERENTIATION_AD_VAR_H_
