#ifndef NUMERICAL_RECIPES_INTEGRATION_QUADRATURE_H_
#define NUMERICAL_RECIPES_INTEGRATION_QUADRATURE_H_

#include <cmath>
#include <vector>
#include <limits>

#include "util/util.h"

namespace NumericalRecipes {
namespace Integration {

// Quadrature integration method interface.
template <typename T>
class Quadrature {
 public:
  using Weights = std::vector<T>;
  using Values = std::vector<T>;

  virtual ~Quadrature() {}

  // Integrates the function f between x_min and x_max.
  T operator()(const std::function<T(T)>& f, T x_min, T x_max) const {
    CHECK(x_min < x_max) << "x_min >= x_max";

    const auto& weights = weightsImpl();
    const auto values = valuesImpl(f, x_min, x_max);

    return std::inner_product(cbegin(weights), cend(weights), cbegin(values),
                              0.0);
  }

  size_t nUnits() const { return nUnitsImpl(); }

 private:
  virtual const Weights& weightsImpl() const = 0;
  virtual Values valuesImpl(const std::function<T(T)>& f, T x_min,
                            T x_max) const = 0;
  virtual size_t nUnitsImpl() const = 0;
};

// Register templates for different quadrature methods.
// Weight are quadrature weights.
// Lambda is the normalized point to the interval (-1, 1) on the x axis.
template <typename T>
class QuadratureTemplate : public Quadrature<T> {
 public:
  using Weights = typename QuadratureTemplate<T>::Weights;
  using Values = typename QuadratureTemplate<T>::Values;
  using WeightTemplate = std::vector<T>;
  using LambdaTemplate = std::vector<T>;

  QuadratureTemplate(size_t nUnits, const WeightTemplate& weightTemplate,
                     const LambdaTemplate& lambdaTemplate)
      : nUnits_(nUnits),
        weightTemplate_(weightTemplate),
        lambdaTemplate_(lambdaTemplate) {
    CHECK(nUnits_ > 0) << "nUnits == 0";
  }
  virtual ~QuadratureTemplate() {}

 private:
  const Weights& weightsImpl() const final;
  Values valuesImpl(const std::function<T(T)>& fn, T x_min,
                    T x_max) const final;
  size_t nUnitsImpl() const final { return nUnits_; }

  size_t nUnits_;
  WeightTemplate weightTemplate_;
  LambdaTemplate lambdaTemplate_;
  mutable Weights weights_cache_;
};

template <typename T>
auto QuadratureTemplate<T>::weightsImpl() const -> const Weights & {
  // TODO(alvin): make this more efficient by not double counting.
  if (weights_cache_.empty()) {
    weights_cache_.reserve(weightTemplate_.size() * this->nUnits());
    for (auto index = 0ul; index < this->nUnits(); ++index) {
      for (const auto& weight : weightTemplate_) {
        weights_cache_.emplace_back(weight);
      }
    }
  }

  return weights_cache_;
}

template <typename T>
auto QuadratureTemplate<T>::valuesImpl(const std::function<T(T)>& f, T x_min,
                                       T x_max) const -> Values {
  const auto h = (x_max - x_min) / this->nUnits();

  // memoize to reduce the number of calls to fn
  auto x_memoized = std::numeric_limits<T>::quiet_NaN();
  auto f_memoized = T(0);

  Values values;
  values.reserve(lambdaTemplate_.size() * this->nUnits());

  // TODO(alvin): make this more efficient by not double counting.
  for (auto index = 0ul; index < this->nUnits(); ++index) {
    const auto x_start = x_min + index * h;
    for (const auto& lambda : lambdaTemplate_) {
      const auto x = x_start + (lambda + 1.0) / 2.0 * h;
      if (!Util::almostEqual(x, x_memoized)) {
        x_memoized = x;
        f_memoized = f(x);
      }
      values.emplace_back(f_memoized * h / 2.0);
    }
  }

  return values;
}

template <typename T>
class Simpson : public QuadratureTemplate<T> {
 public:
  explicit Simpson(size_t nUnits)
      : QuadratureTemplate<T>(nUnits, {1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0},
                              {-1.0, 0, 1.0}) {}
};

template <typename T>
class GaussLobatto4 : public QuadratureTemplate<T> {
 public:
  explicit GaussLobatto4(size_t nUnits)
      : QuadratureTemplate<T>(nUnits,
                              {1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0},
                              {-1.0, -sqrt(1.0 / 5.0), sqrt(1.0 / 5.0), 1.0}) {}
};

template <typename T>
class GaussLobatto5 : public QuadratureTemplate<T> {
 public:
  explicit GaussLobatto5(size_t nUnits)
      : QuadratureTemplate<T>(
            nUnits,
            {1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0},
            {-1.0, -sqrt(3.0 / 7.0), 0, sqrt(3.0 / 7.0), 1.0}) {}
};

template <typename T>
class GaussLegendre3 : public QuadratureTemplate<T> {
 public:
  explicit GaussLegendre3(size_t nUnits)
      : QuadratureTemplate<T>(nUnits, {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0},
                              {-sqrt(3.0 / 5.0), 0, sqrt(3.0 / 5.0)}) {}
};

template <typename T>
class GaussLegendre4 : public QuadratureTemplate<T> {
 public:
  explicit GaussLegendre4(size_t nUnits)
      : QuadratureTemplate<T>(
            nUnits, {(18.0 + sqrt(30.0)) / 36.0, (18.0 + sqrt(30.0)) / 36.0,
                     (18.0 - sqrt(30.0)) / 36.0, (18.0 - sqrt(30.0)) / 36.0},
            {sqrt(3.0 / 7.0 - 2.0 / 7.0 * sqrt(6.0 / 5.0)),
             -sqrt(3.0 / 7.0 - 2.0 / 7.0 * sqrt(6.0 / 5.0)),
             sqrt(3.0 / 7.0 + 2.0 / 7.0 * sqrt(6.0 / 5.0)),
             -sqrt(3.0 / 7.0 + 2.0 / 7.0 * sqrt(6.0 / 5.0))}) {}
};

}  // namespace Integration
}  // namespace NumericalRecipes

#endif  // NUMERICAL_RECIPES_INTEGRATION_QUADRATURE_H_
