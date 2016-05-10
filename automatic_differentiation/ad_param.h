#ifndef AUTOMATIC_DIFFERENTIATION_AD_PARAM_H_
#define AUTOMATIC_DIFFERENTIATION_AD_PARAM_H_

#include <locale>
#include <memory>
#include <string>

#include "automatic_differentiation/ad.h"
#include "automatic_differentiation/ad_const.h"
#include "util/clonable.h"

namespace AutomaticDifferentiation {

class AD::Param : public Expression {
 public:
  using VarValues = AD::VarValues;

  static std::unique_ptr<Expression> make(const std::string& identifier,
                                          double value) {
    return Param(identifier, value).clone();
  }
  Param(const Param&) = default;
  Param& operator=(const Param&) = default;
  virtual ~Param() {}

  const std::string& identifier() const { return identifier_; }
  double value() const { return value_; }
  double& value() { return value_; }

 private:
  explicit Param(const std::string& identifier, double value)
      : identifier_(identifier), value_(value) {}

  AD differentiateImpl(const AD& var) const final {
    return AD(identifier() == AutomaticDifferentiation::identifier(var) ? 1
                                                                        : 0);
  }

  AD evaluateAtImpl(const VarValues& /*varValues*/) const final {
    return AD(identifier(), value());
  }

  AD simplifyImpl() const final { return AD(identifier(), value()); }

  std::string expressionImpl() const final { return identifier(); }

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Param>(*this);
  }

  std::string identifier_;
  double value_;
};

}  // namespace AutomaticDifferentiation

#endif  // AUTOMATIC_DIFFERENTIATION_AD_PARAM_H_
