#include "integration/quadrature.h"

#include <iostream>

#include "util/util.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

TEST(Quadrature, Simpson) {
  using NumericalRecipes::Integration::Simpson;

  Simpson<double> integrate(20);

  auto epsilon = 1E-6;
  auto result1 =
      integrate([](double x) { return std::sin(x); }, -M_PI_2, M_PI_2);
  EXPECT_NEAR(result1, 0, epsilon);

  auto result2 =
      integrate([](double x) { return std::cos(x); }, -M_PI_2, M_PI_2);
  EXPECT_NEAR(result2, 2, epsilon);
}

TEST(Quadrature, GaussLobatto4) {
  using NumericalRecipes::Integration::GaussLobatto4;

  GaussLobatto4<double> integrate(20);

  auto epsilon = 1E-6;
  auto result1 =
      integrate([](double x) { return std::sin(x); }, -M_PI_2, M_PI_2);
  EXPECT_NEAR(result1, 0, epsilon);

  auto result2 =
      integrate([](double x) { return std::cos(x); }, -M_PI_2, M_PI_2);
  EXPECT_NEAR(result2, 2, epsilon);
}

TEST(Quadrature, GaussLobatto5) {
  using NumericalRecipes::Integration::GaussLobatto5;

  GaussLobatto5<double> integrate(20);

  auto epsilon = 1E-6;
  auto result1 =
      integrate([](double x) { return std::sin(x); }, -M_PI_2, M_PI_2);
  EXPECT_NEAR(result1, 0, epsilon);

  auto result2 =
      integrate([](double x) { return std::cos(x); }, -M_PI_2, M_PI_2);
  EXPECT_NEAR(result2, 2, epsilon);
}

TEST(Quadrature, GaussLegendre3) {
  using NumericalRecipes::Integration::GaussLegendre3;

  GaussLegendre3<double> integrate(20);

  auto epsilon = 1E-6;
  auto result1 =
      integrate([](double x) { return std::sin(x); }, -M_PI_2, M_PI_2);
  EXPECT_NEAR(result1, 0, epsilon);

  auto result2 =
      integrate([](double x) { return std::cos(x); }, -M_PI_2, M_PI_2);
  EXPECT_NEAR(result2, 2, epsilon);
}

TEST(Quadrature, GaussLegendre4) {
  using NumericalRecipes::Integration::GaussLegendre4;

  GaussLegendre4<double> integrate(20);

  auto epsilon = 1E-6;
  auto result1 =
      integrate([](double x) { return std::sin(x); }, -M_PI_2, M_PI_2);
  EXPECT_NEAR(result1, 0, epsilon);

  auto result2 =
      integrate([](double x) { return std::cos(x); }, -M_PI_2, M_PI_2);
  EXPECT_NEAR(result2, 2, epsilon);
}
