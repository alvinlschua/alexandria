#ifndef AUTOMATIC_DIFFERENTIATION_AD_EXPRESSION_H
#define AUTOMATIC_DIFFERENTIATION_AD_EXPRESSION_H

#include <memory>
#include <string>

#include "automatic_differentiation/ad.h"
#include "util/clonable.h"

namespace AutomaticDifferentiation {

// Abstract expression class.
class AD::Expression : public Util::Clonable<AD::Expression> {
 public:
  using VarValues = AD::VarValues;

  virtual ~Expression() {}
  Expression() = default;
  Expression(const Expression&) = default;
  Expression& operator=(const Expression&) = default;

  // Differentiate with respect to AD::Var.
  AD differentiate(const AD& var) const;

  // Evaluate the expression with concrete values for AD::Var.
  AD evaluateAt(const VarValues& varValues) const;

  // Simplify the (sub) expression.
  AD simplify() const { return simplifyImpl(); }

  // Get the expression as a string.
  std::string expression() const { return expressionImpl(); }

 private:
  virtual AD differentiateImpl(const AD& var) const = 0;
  virtual AD evaluateAtImpl(const VarValues& varValues) const = 0;
  virtual AD simplifyImpl() const = 0;
  virtual std::string expressionImpl() const = 0;
};

}  // namespace AutomaticDifferntiation

#endif
