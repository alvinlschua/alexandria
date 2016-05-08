#ifndef AUTOMATIC_DIFFERENTIATION_AD_VAR_H
#define AUTOMATIC_DIFFERENTIATION_AD_VAR_H

#include <locale>
#include <memory>
#include <string>

#include "automatic_differentiation/ad.h"
#include "automatic_differentiation/ad_const.h"
#include "util/clonable.h"

namespace AutomaticDifferentiation {

class AD::Var : public Expression {
 public:
  using VarValues = AD::VarValues;

  static std::unique_ptr<Expression> make(const std::string& identifier) {
    return Var(identifier).clone();
  }
  Var(const Var&) = default;
  Var& operator=(const Var&) = default;

  virtual ~Var() {}

  const std::string& identifier() const { return identifier_; }

 private:
  Var(const std::string& identifier) : identifier_(identifier) {}

  AD differentiateImpl(const AD& var) const final {
    namespace ADiff = ::AutomaticDifferentiation;
    return AD(identifier() == ADiff::identifier(var) ? 1 : 0);
  }

  AD evaluateAtImpl(const VarValues& varValues) const final {
    for (const auto& varValue : varValues) {
      if (expression() == varValue.first.expression()) {
        return AD(varValue.second);
      }
    }
    return AD(identifier());
  }

  AD simplifyImpl() const final { return AD(identifier()); }

  std::string expressionImpl() const final { return identifier(); }

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Var>(*this);
  }

  std::string identifier_;
};

}  // namespace AutomaticDifferntiation

#endif
