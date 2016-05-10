#include <cmath>
#include "automatic_differentiation/ad.h"
#include "automatic_differentiation/ad_binary.h"
#include "automatic_differentiation/ad_const.h"
#include "automatic_differentiation/ad_var.h"
#include "automatic_differentiation/ad_param.h"
#include "automatic_differentiation/ad_unary.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

TEST(AD, Basic) {
  using AD = AutomaticDifferentiation::AD<double>;
  AD c(5.0);

  AD x("x");
  auto y = AD("y");
  auto z = y;

  EXPECT_EQ(c.expression(), "5.000000");
  EXPECT_TRUE(c.isType<AD::Const>());
  EXPECT_FALSE(c.isType<AD::Var>());
  EXPECT_EQ(c.evaluateAt({x = 2}).expression(), "5.000000");
  EXPECT_EQ(c.differentiate(x).expression(), "0.000000");
  EXPECT_TRUE(c.differentiate(x).isType<AD::Const>());
  EXPECT_EQ(value(c), 5);

  EXPECT_EQ(x.expression(), "x");
  EXPECT_TRUE(x.isType<AD::Var>());
  EXPECT_FALSE(x.isType<AD::Const>());
  EXPECT_EQ(x.evaluateAt({x = 2}).expression(), "2.000000");
  EXPECT_EQ(x.differentiate(x).expression(), "1.000000");
  EXPECT_EQ(x.differentiate(y).expression(), "0.000000");
  EXPECT_EQ(identifier(x), "x");

  EXPECT_EQ(y.expression(), "y");
  EXPECT_TRUE(y.isType<AD::Var>());
  EXPECT_FALSE(y.isType<AD::Const>());
  EXPECT_EQ(y.evaluateAt({x = 2}).expression(), "y");
  EXPECT_EQ(y.differentiate(x).expression(), "0.000000");
  EXPECT_EQ(y.differentiate(y).expression(), "1.000000");

  EXPECT_EQ(z.expression(), "y");
  EXPECT_TRUE(z.isType<AD::Var>());
  EXPECT_FALSE(z.isType<AD::Const>());
  EXPECT_EQ(z.evaluateAt({x = 2}).expression(), "y");
  EXPECT_EQ(z.differentiate(x).expression(), "0.000000");
  EXPECT_EQ(z.differentiate(y).expression(), "1.000000");
}

TEST(AD, ConstAndVar) {
  using AD = AutomaticDifferentiation::AD<double>;
  auto c = AD(5.0);
  auto x = AD("x");
  auto y = AD("y");

  EXPECT_EQ(value(D(c, x)), 0);
  EXPECT_EQ(value(D(x, x)), 1);
  EXPECT_EQ(value(D(y, x)), 0);

  EXPECT_EQ(value(D(c, {x})), 0);
  EXPECT_EQ(value(D(x, {x})), 1);
  EXPECT_EQ(value(D(x, {x, x})), 0);
  EXPECT_EQ(value(D(y, {x})), 0);
}

TEST(AD, Exceptions) {
  using AD = AutomaticDifferentiation::AD<double>;
  auto c = AD(5.0);
  auto x = AD("x");
  auto y = AD("y");

  EXPECT_THROW(AD("1x"), std::invalid_argument);
  EXPECT_THROW(AD(""), std::invalid_argument);
  EXPECT_THROW(AD("abcdefghi"), std::invalid_argument);
  EXPECT_NO_THROW(AD("abcdefgh"));

  EXPECT_THROW(value(x), std::logic_error);
  EXPECT_THROW(identifier(c), std::logic_error);
  //EXPECT_THROW(value(x + y), std::logic_error);
  //EXPECT_THROW(identifier(x + y), std::logic_error);

  EXPECT_NO_THROW(value(c));
  EXPECT_NO_THROW(identifier(x));
}

TEST(AD, Op) {
  using AD = AutomaticDifferentiation::AD<double>;
  auto x = AD("x");
  auto y = AD("y");
  auto z = AD("z");

  EXPECT_EQ(value(D(x + y, x)), 1);
  EXPECT_EQ(value(D(x + y, y)), 1);
  EXPECT_EQ(value(D(x + y, z)), 0);
  EXPECT_EQ((x + y).evaluateAt({x = 3}).expression(), "(3.000000 + y)");

  EXPECT_EQ(value(D(5.0 + y, y)), 1);
  EXPECT_EQ(value(D(y + 5.0, y)), 1);
  EXPECT_EQ(value(D(5.0 - y, y)), -1);
  EXPECT_EQ(value(D(y - 5.0, y)), 1);

  EXPECT_EQ(value(D(x + y, x)), 1);
  EXPECT_EQ(value(D(x + y, y)), 1);
  EXPECT_EQ(value(D(x - y, x)), 1);
  EXPECT_EQ(value(D(x - y, y)), -1);

  EXPECT_EQ(value(D(x + x - y, x)), 2);
  EXPECT_EQ(value(D(x + x - y, y)), -1);
  EXPECT_EQ(value(D(x - y - y + x, x)), 2);
  EXPECT_EQ(value(D(x - y - y + x, y)), -2);

  EXPECT_EQ(value(D(x * x, x).evaluateAt({x = 2})), 4);
  EXPECT_EQ(value(D(3.0 * x * x + 5.0, x).evaluateAt({x = 2})), 12);

  EXPECT_EQ(value(D(x * y, x).evaluateAt({x = 3, y = 4})), 4);
  EXPECT_EQ(value(D(x * y, y).evaluateAt({x = 3, y = 4})), 3);
  EXPECT_EQ(value(D(x / y, x).evaluateAt({x = 3, y = 4})), 1.0 / 4.0);
  EXPECT_EQ(value(D(x / y, y).evaluateAt({x = 3, y = 4})), -3.0 / 16.0);

  EXPECT_EQ(value(D(x * (x + y), x).evaluateAt({x = 3, y = 4})), 10);

  EXPECT_EQ(value(D(sin(y), y).evaluateAt({y = 1})), cos(1.0));
  EXPECT_EQ(value(D(cos(y), y).evaluateAt({y = 1})), -sin(1.0));

  EXPECT_EQ(value(D(cos(x * x), x).evaluateAt({x = 1})), -2.0 * sin(1.0));

  EXPECT_EQ(value(D(exp(-x * x), x).evaluateAt({x = 1})), -2.0 * exp(-1.0));
  EXPECT_EQ(value(D(log(x), x).evaluateAt({x = 1})), 1.0);
  EXPECT_EQ(D(pow(x, 2.0), x).expression(), "2.000000 * x");
  EXPECT_EQ(D(0.5 * pow(x, 2.0), x).expression(), "x");
  EXPECT_EQ(D(pow(x, 1.0), x).expression(), "1.000000");
  EXPECT_EQ(value(D(pow(5.0, x), x).evaluateAt({x = 1})), 5.0 * log(5.0));
  EXPECT_EQ(value(D(pow(x, 2.0), x).evaluateAt({x = 1})), 2.0);

  auto g = grad(sin(x * y), {x, y});
  EXPECT_EQ(value(g[0].evaluateAt({x = 1, y = 2})), 2.0 * cos(2.0));
  EXPECT_EQ(value(g[1].evaluateAt({x = 1, y = 2})), 1.0 * cos(2.0));
}

/*
TEST(AD, Param) {
  using AutomaticDifferentiation::AD;
  auto x = AD("x");
  auto y = AD("y");
  auto c = AD("c", 5);

  EXPECT_EQ(value(D(c * x, x)), 5);
  c.reference<AD::Param>().value() = 6;

  EXPECT_EQ(value(D(c * x, x)), 6);
  EXPECT_EQ(value(D(c * x, c).evaluateAt({x = 1})), 1);
}
*/
