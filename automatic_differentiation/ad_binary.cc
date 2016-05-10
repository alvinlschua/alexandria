#include "automatic_differentiation/ad_binary.h"

#include <cmath>
#include <utility>

#include "automatic_differentiation/ad_unary.h"
#include "util/util.h"

namespace AutomaticDifferentiation {

AD AD::Binary::differentiateImpl(const AD& var) const {
  // TODO(alvin) Only reverse mode at the moment. Consider implementing forward
  // mode.
  auto result = dF1() * adFirst().differentiate(var) +
                dF2() * adSecond().differentiate(var);

  return result.simplify();
}

AD AD::Binary::evaluateAtImpl(const VarValues& varValues) const {
  auto ad1 = (!adFirst().isType<Const>())
                 ? adFirst().evaluateAt(varValues)
                 : adFirst();
  auto ad2 = (!adSecond().isType<Const>())
                 ? adSecond().evaluateAt(varValues)
                 : adSecond();

  ad1 = ad1.simplify();
  ad2 = ad2.simplify();

  if (ad1.isType<Const>() && ad2.isType<Const>()) {
    return AD(f(value(ad1), value(ad2)));
  }

  auto ptr = clone();
  dynamic_cast<Binary*>(ptr.get())->ad1_ = ad1;
  dynamic_cast<Binary*>(ptr.get())->ad2_ = ad2;

  return AD(std::move(ptr)).simplify();
}

// Plus
double Plus::f(double value1, double value2) const { return value1 + value2; }
AD Plus::dF1() const { return AD(1.0); }
AD Plus::dF2() const { return AD(1.0); }

AD Plus::simplifyImpl() const {
  using Const = AD::Const;

  auto ad1 = adFirst().simplify();
  auto ad2 = adSecond().simplify();

  if (ad1.isType<Const>() && ad2.isType<Const>()) {
    return AD(f(value(ad1), value(ad2)));
  }

  // Convert to Const + Expression if possible
  if (ad2.isType<Const>()) {
    std::swap(ad1, ad2);
  }

  // 0 + Expression -> -Expression
  if (ad1.isType<Const>() && Util::almostEqual(value(ad1), 0)) {
    return ad2;
  }

  // (c1 * Expression) + (c2 * Expression) = (c1 + c2) * Expression
  if (ad1.isType<Times>() && ad2.isType<Times>()) {
    auto& ad11 = ad1.reference<Times>().adFirst();
    auto& ad12 = ad1.reference<Times>().adSecond();
    auto& ad21 = ad2.reference<Times>().adFirst();
    auto& ad22 = ad2.reference<Times>().adSecond();

    if (ad11.isType<Const>() && ad21.isType<Const>() &&
        (ad12.expression() == ad22.expression())) {
      auto result = (ad11 + ad21) * ad12;
      return result.simplify();
    }
  }

  return ad1 + ad2;
}

AD operator+(const AD& ad1, const AD& ad2) { return Plus::makeAD(ad1, ad2); }

AD operator+(double value, const AD& ad2) {
  return Plus::makeAD(AD(value), ad2);
}

AD operator+(const AD& ad1, double value) {
  return Plus::makeAD(ad1, AD(value));
}

// Minus
double Minus::f(double value1, double value2) const { return value1 - value2; }
AD Minus::dF1() const { return AD(1.0); }
AD Minus::dF2() const { return AD(-1.0); }

AD Minus::simplifyImpl() const {
  auto ad1 = adFirst().simplify();
  auto ad2 = adSecond().simplify();

  if (ad1.isType<AD::Const>() && ad2.isType<AD::Const>()) {
    return AD(f(value(ad1), value(ad2)));
  }

  // Convert to AD::Const - Expression if possible
  if (ad2.isType<AD::Const>()) {
    std::swap(ad1, ad2);
  }

  // 0 - Expression -> -Expression
  if (ad1.isType<AD::Const>() && Util::almostEqual(value(ad1), 0)) {
    return -ad2;
  }

  // (c1 * Expression) + (c2 * Expression) = (c1 + c2) * Expression
  if (ad1.isType<Times>() && ad2.isType<Times>()) {
    auto& ad11 = ad1.reference<Times>().adFirst();
    auto& ad12 = ad1.reference<Times>().adSecond();
    auto& ad21 = ad2.reference<Times>().adFirst();
    auto& ad22 = ad2.reference<Times>().adSecond();

    if (ad11.isType<AD::Const>() && ad21.isType<AD::Const>() &&
        (ad12.expression() == ad22.expression())) {
      auto result = (ad11 + ad21) * ad12;
      return result.simplify();
    }
  }

  return ad1 - ad2;
}

AD operator-(const AD& ad1, const AD& ad2) { return Minus::makeAD(ad1, ad2); }

AD operator-(double value, const AD& ad2) {
  return Minus::makeAD(AD(value), ad2);
}

AD operator-(const AD& ad1, double value) {
  return Minus::makeAD(ad1, AD(value));
}

// Times
double Times::f(double value1, double value2) const { return value1 * value2; }

AD Times::dF1() const { return adSecond(); }

AD Times::dF2() const { return adFirst(); }

AD Times::simplifyImpl() const {
  auto ad1 = adFirst().simplify();
  auto ad2 = adSecond().simplify();

  if (ad1.isType<AD::Const>() && ad2.isType<AD::Const>()) {
    return AD(f(value(ad1), value(ad2)));
  }

  // Convert to AD::Const * Expression if possible
  if (ad2.isType<AD::Const>()) {
    std::swap(ad1, ad2);
  }

  // 0 * Exression -> 0
  if (ad1.isType<AD::Const>() && Util::almostEqual(value(ad1), 0)) {
    return AD(0);
  }

  // 1 * Exression -> Expression
  if (ad1.isType<AD::Const>() && Util::almostEqual(value(ad1), 1)) {
    return ad2;
  }

  if (ad2.isType<Times>()) {
    // AD::Const * (AD::Const * Expression) -> AD::Const * expression
    auto ad2Ptr = ad2.pointer<Times>();
    if (ad1.isType<AD::Const>() && ad2Ptr->adFirst().isType<AD::Const>()) {
      auto result =
          AD(f(value(ad1), value(ad2Ptr->adFirst()))) * ad2Ptr->adSecond();
      return result.simplify();
    }
  }

  return ad1 * ad2;
}

AD operator*(const AD& ad1, const AD& ad2) { return Times::makeAD(ad1, ad2); }

AD operator*(double value, const AD& ad2) {
  return Times::makeAD(AD(value), ad2);
}

AD operator*(const AD& ad1, double value) {
  return Times::makeAD(ad1, AD(value));
}

// Divide
double Divide::f(double value1, double value2) const { return value1 / value2; }

AD Divide::dF1() const { return 1.0 / adSecond(); }

AD Divide::dF2() const { return -adFirst() / (adSecond() * adSecond()); }

AD Divide::simplifyImpl() const {
  auto ad1 = adFirst().simplify();
  auto ad2 = adSecond().simplify();

  if (ad1.isType<AD::Const>() && ad2.isType<AD::Const>()) {
    return AD(f(value(ad1), value(ad2)));
  }

  // Expression / AD::Const -> AD::Const * Expression;
  if (ad2.isType<AD::Const>()) {
    return AD(1.0 / value(ad2)) * ad1;
  }

  return ad1 / ad2;
}

AD operator/(const AD& ad1, const AD& ad2) { return Divide::makeAD(ad1, ad2); }

AD operator/(double value, const AD& ad2) {
  return Divide::makeAD(AD(value), ad2);
}

AD operator/(const AD& ad1, double value) {
  return Divide::makeAD(ad1, AD(value));
}

// Pow
double Pow::f(double value1, double value2) const {
  return std::pow(value1, value2);
}

AD Pow::dF1() const { return adSecond() * pow(adFirst(), adSecond() - 1); }

AD Pow::dF2() const { return pow(adFirst(), adSecond()) * log(adFirst()); }

AD Pow::simplifyImpl() const {
  auto ad1 = adFirst().simplify();
  auto ad2 = adSecond().simplify();

  if (ad1.isType<AD::Const>() && ad2.isType<AD::Const>()) {
    return AD(f(value(ad1), value(ad2)));
  }

  // Expression / AD::Const -> AD::Const * Expression;
  if (ad2.isType<AD::Const>()) {
    if (Util::almostEqual(value(ad2), 1)) {
      return ad1;
    }
    if (Util::almostEqual(value(ad2), 0)) {
      return AD(1);
    }
  }

  return pow(ad1, ad2);
}

AD pow(const AD& ad1, const AD& ad2) { return Pow::makeAD(ad1, ad2); }
AD pow(const AD& ad, double value) { return Pow::makeAD(ad, AD(value)); }
AD pow(double value, const AD& ad) { return Pow::makeAD(AD(value), ad); }

}  // namespace AutomaticDifferentiation
