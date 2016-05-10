#ifndef AUTOMATIC_DIFFERENTIATION_AD_H_
#define AUTOMATIC_DIFFERENTIATION_AD_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "glog/logging.h"

namespace AutomaticDifferentiation {

// A wrapper class that implements a simple for of type erasure.
class AD {
 public:
  class Expression;
  class Const;
  class Param;
  class Var;
  class Unary;
  class Binary;

  using VarValue = std::pair<const AD&, double>;
  using VarValues = std::vector<VarValue>;
  using Ptr = std::unique_ptr<Expression>;

  // Default constructor.
  AD() {}

  // Make constant.
  explicit AD(double value);

  // Make var.
  explicit AD(const std::string& identifier);

  // Make param. Params can be treated as variables, but have an associated set
  // of values.
  explicit AD(const std::string& identifier, double value);

  // Construct from a ptr.
  explicit AD(Ptr ptr) : ptr_(std::move(ptr)) {}

  // Copy constructor.
  AD(const AD& ad);

  // Copy assignment.
  AD& operator=(const AD& ad);

  // Produce a VarValue when assigning double to a var.  Not that this is not a
  // constructor, but is implemented for notation.
  AD::VarValue operator=(double value);

  // Differentiate with respect to AD::Var.
  AD differentiate(const AD& var) const;

  // Evaluate the expression with concrete values for AD::Var.
  AD evaluateAt(const VarValues& varValues) const;

  // Simplify the expression.
  AD simplify() const;

  // Get the expression as a string.
  std::string expression() const;

  // Is this of type T?
  template <typename T>
  bool isType() const {
    return pointer<T>() != nullptr;
  }

  // Cast this to type T.  A nullptr is return is it cannot be cast.
  template <typename T>
  T* pointer() const {
    return dynamic_cast<T*>(ptr_.get());
  }

  // Cast this to reference of type T.  This fails if isType<T> is false.
  template <typename T>
  T& reference() const {
    auto result = pointer<T>();
    if (result == nullptr) {
      throw std::logic_error(std::string("unable to cast to ") +
                             typeid(T).name() + "in castReference");
    }
    return *result;
  }

 private:
  Ptr ptr_;
};

// Gets the value when ad is a Const.
double value(const AD& ad);

// Gets the identifier when ad is a Var.
std::string identifier(const AD& ad);

// Differentiate.
inline AD D(const AD& expr, const AD& var) { return expr.differentiate(var); }

// Differentiate, multivariate.
AD D(const AD& expr, const std::vector<AD>& vars);

// TODO(alvin) Make ADVector its own class and provide methods like evaluateAt
// and simplify.
using ADVector = std::vector<AD>;

// Grad operator.
ADVector grad(const AD& expr, const std::vector<AD>& vars);

}  // namespace AutomaticDifferentiation

#endif  // AUTOMATIC_DIFFERENTIATION_AD_H_
