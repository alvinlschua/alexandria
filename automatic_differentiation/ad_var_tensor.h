#ifndef AUTOMATIC_DIFFERENTIATION_AD_VAR_TENSOR_H_
#define AUTOMATIC_DIFFERENTIATION_AD_VAR_TENSOR_H_

#include "automatic_differentiation/ad_expression_tensor.h"
#include "tensor/tensor.h"

namespace Alexandria {

template <typename T>
class AD<T>::Var : public Expression {
 public:
  using VarValues = AD<T>::VarValues;

  static std::unique_ptr<Expression> make(const std::string& identifier,
                                          const Shape& shape) {
    return Var(identifier, shape).clone();
  }
  Var(const Var&) = default;
  Var& operator=(const Var&) = default;

  virtual ~Var() {}

  const std::string& identifier() const { return identifier_; }

 private:
  explicit Var(const std::string& identifier, const Shape& shape)
      : identifier_(identifier), shape_(shape) {}

  AD<T> differentiateImpl(const AD<T>& var) const final {
    return AD<T>(identifier() == Alexandria::identifier<T>(var)
                     ? T::sparseEye(combineShapes(this->shape(), this->shape()))
                     : T::sparse(combineShapes(this->shape(), var.shape())));
  }

  AD<T> evaluateAtImpl(const VarValues& varValues) const final {
    for (const auto& varValue : varValues) {
      if (identifier() == Alexandria::identifier<T>(varValue.first)) {
        if (varValue.second.shape() != shape_) {
          throw std::invalid_argument("Shape of tensor provided for variable " +
                                      identifier() + " does not match");
        }
        return AD<T>(varValue.second);
      }
    }
    return AD<T>(identifier(), this->shape());
  }

  AD<T> simplifyImpl() const final {
    return AD<T>(identifier(), this->shape());
  }

  const Shape& shapeImpl() const { return shape_; }

  std::string expressionImpl() const final { return identifier(); }

  std::unique_ptr<Expression> cloneImpl() const {
    return std::make_unique<Var>(*this);
  }

  std::string identifier_;
  Shape shape_;
};

}  // namespace Alexandria

#endif  // AUTOMATIC_DIFFERENTIATION_AD_VAR_TENSOR_H_
