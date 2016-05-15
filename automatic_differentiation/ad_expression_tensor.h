#ifndef AUTOMATIC_DIFFERENTIATION_AD_EXPRESSION_TENSOR_H_
#define AUTOMATIC_DIFFERENTIATION_AD_EXPRESSION_TENSOR_H_

#include <memory>
#include <string>

#include "automatic_differentiation/ad_tensor.h"
#include "util/clonable.h"

namespace AutomaticDifferentiation {

// Abstract expression class.
template <typename T>
class AD<T>::Expression : public Util::Clonable<AD<T>::Expression> {
 public:
  using VarValues = AD<T>::VarValues;

  virtual ~Expression() {}
  Expression() = default;
  Expression(const Expression&) = default;
  Expression& operator=(const Expression&) = default;

  // Differentiate with respect to AD<T>::Var.
  AD<T> differentiate(const AD<T>& var) const {
    CHECK(var.isType<typename AD<T>::Var>() ||
          var.isType<typename AD<T>::Param>())
        << "must be of type Var or Param";
    return differentiateImpl(var);
  }

  // Evaluate the expression with concrete values for AD<T>::Var.
  AD<T> evaluateAt(const VarValues& varValues) const {
    for (const auto& varValue : varValues) {
      CHECK(varValue.first.template isType<typename AD<T>::Var>())
          << "must be of type Var";
    }
    return evaluateAtImpl(varValues);
  }

  // Simplify the (sub) expression.
  AD<T> simplify() const { return simplifyImpl(); }

  // Shape of the result.
  const Shape& shape() const { return shapeImpl(); }

  // Get the expression as a string.
  std::string expression() const { return expressionImpl(); }

 private:
  virtual AD<T> differentiateImpl(const AD<T>& var) const = 0;
  virtual AD<T> evaluateAtImpl(const VarValues& varValues) const = 0;
  virtual AD<T> simplifyImpl() const = 0;
  virtual const Shape& shapeImpl() const = 0;
  virtual std::string expressionImpl() const = 0;
};

}  // namespace AutomaticDifferentiation

#endif  //  AUTOMATIC_DIFFERENTIATION_AD_EXPRESSION_TENSOR_H_
