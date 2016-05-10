#include "automatic_differentiation/ad_expression.h"
#include "automatic_differentiation/ad_var.h"

namespace AutomaticDifferentiation {

AD AD::Expression::differentiate(const AD& var) const {
  CHECK(var.isType<AD::Var>()) << "must be of type Var";
  return differentiateImpl(var);
}

// Evaluate the expression with concrete values for AD::Var.
AD AD::Expression::evaluateAt(const VarValues& varValues) const {
  for (const auto& varValue : varValues) {
    CHECK(varValue.first.isType<AD::Var>()) << "must be of type Var";
  }
  return evaluateAtImpl(varValues);
}
}  // namespace AutomaticDifferentiation
