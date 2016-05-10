#include "automatic_differentiation/ad.h"
#include "automatic_differentiation/ad_const.h"
#include "automatic_differentiation/ad_var.h"

#include "util/util.h"

namespace AutomaticDifferentiation {

AD::AD(double value) : ptr_(Const::make(value)) {}
AD::AD(const std::string& identifier) : ptr_(Var::make(identifier)) {
  if (identifier.size() == 0)
    throw std::invalid_argument("identifier should be specified");

  if (identifier.size() > 8)
    throw std::invalid_argument("identifier should have at most 8 characters");

  std::locale loc;
  if (!std::isalpha(identifier[0], loc))
    throw std::invalid_argument("identifier should start with a letter");
}

AD::AD(const AD& ad) { ptr_ = std::move(ad.ptr_->clone()); }

AD& AD::operator=(const AD& ad) {
  ptr_ = std::move(ad.ptr_->clone());
  return *this;
}

AD::VarValue AD::operator=(double value) {
  CHECK(isType<AD::Var>()) << "should only assign doubles to a Var type";
  return AD::VarValue(*this, value);
}

AD AD::differentiate(const AD& var) const { return ptr_->differentiate(var); }

AD AD::evaluateAt(const VarValues& varValues) const {
  return ptr_->evaluateAt(varValues);
}

AD AD::simplify() const { return ptr_->simplify(); }

std::string AD::expression() const { return ptr_->expression(); }

double value(const AD& ad) { return ad.reference<AD::Const>().value(); }

std::string identifier(const AD& ad) {
  return ad.reference<AD::Var>().identifier();
}

AD D(const AD& expr, const std::vector<AD>& vars) {
  AD result(expr);
  for (const auto& var : vars) {
    result = D(result, var);
  }
  return result;
}

ADVector grad(const AD& expr, const std::vector<AD>& vars) {
  ADVector result;
  result.reserve(vars.size());
  for (const auto& var : vars) {
    result.emplace_back(D(expr, var));
  }
  return result;
}

}  // namespace AutomaticDifferentiation
