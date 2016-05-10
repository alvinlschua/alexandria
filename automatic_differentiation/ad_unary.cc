#include <cmath>
#include "automatic_differentiation/ad_const.h"
#include "automatic_differentiation/ad_binary.h"
#include "automatic_differentiation/ad_unary.h"

namespace AutomaticDifferentiation {

AD AD::Unary::differentiateImpl(const AD& var) const {
  // TODO(alvin) Only reverse mode at the moment. Consider implementing forward
  // mode.
  auto result = dF() * ad_.differentiate(var);
  return result.simplify();
}

AD AD::Unary::evaluateAtImpl(const VarValues& varValues) const {
  auto ad = !ad_.isType<Const>() ? ad_.evaluateAt(varValues) : ad_;
  ad = ad.simplify();

  if (ad.isType<Const>()) {
    return AD(f(ad.reference<Const>().value()));
  }

  auto ptr = clone();
  dynamic_cast<Unary*>(ptr.get())->ad_ = ad;

  return AD(std::move(ptr));
}

AD AD::Unary::simplifyImpl() const {
  return ad_.isType<Const>() ? AD(f(value(ad_))) : AD(clone());
}

double UnaryMinus::f(double value) const { return -value; }

AD UnaryMinus::dF() const { return AD(-1); }

AD UnaryMinus::simplifyImpl() const {
  AD result;
  if (ad_.isType<AD::Const>()) {
    result = AD(f(value(ad_)));
  } else if (ad_.isType<UnaryMinus>()) {
    result = ad_.reference<UnaryMinus>().ad_;
  } else {
    result = AD(clone());
  }
  return result;
}

AD operator-(const AD& ad) { return UnaryMinus::makeAD(ad); }

double Sin::f(double value) const { return std::sin(value); }
AD Sin::dF() const { return cos(ad_); }
AD sin(const AD& ad) { return Sin::makeAD(ad); }

double Cos::f(double value) const { return std::cos(value); }
AD Cos::dF() const { return -sin(ad_); }
AD cos(const AD& ad) { return Cos::makeAD(ad); }

double Exp::f(double value) const { return std::exp(value); }
AD Exp::dF() const { return exp(ad_); }
AD exp(const AD& ad) { return Exp::makeAD(ad); }

double Log::f(double value) const { return std::log(value); }
AD Log::dF() const { return 1.0 / ad_; }
AD log(const AD& ad) { return Log::makeAD(ad); }

}  // namespace AutomaticDifferentiation
