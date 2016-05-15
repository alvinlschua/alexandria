#include <cmath>

#include "automatic_differentiation/ad_binary_tensor.h"
#include "automatic_differentiation/ad_const_tensor.h"
#include "automatic_differentiation/ad_param_tensor.h"
#include "automatic_differentiation/ad_tensor.h"
#include "automatic_differentiation/ad_unary_tensor.h"
#include "automatic_differentiation/ad_var_tensor.h"
#include "neural_net/tensor/helpers.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

TEST(AD, Basic) {
  using TensorDouble = NeuralNet::Tensor<double>;
  using Shape = NeuralNet::Shape;
  using AD = AutomaticDifferentiation::AD<TensorDouble>;

  AD c(TensorDouble({5.0, 2.0, 3.0}));

  AD x("x", Shape({3}));
  auto y = AD("y", Shape({3}));
  auto z = y;

  EXPECT_EQ(c.expression(), "Shape(3){5 2 3}");
  EXPECT_TRUE(c.isType<AD::Const>());
  EXPECT_FALSE(c.isType<AD::Var>());
  EXPECT_EQ(c.evaluateAt({x = TensorDouble({2, 2, 3})}).expression(),
            "Shape(3){5 2 3}");
  EXPECT_EQ(c.differentiate(x).expression(), "Shape(3, 3){0 0 0 0 0 0 0 0 0}");
  EXPECT_TRUE(c.differentiate(x).isType<AD::Const>());
  EXPECT_EQ(value(c), TensorDouble({5, 2, 3}));

  EXPECT_EQ(x.expression(), "x");
  EXPECT_TRUE(x.isType<AD::Var>());
  EXPECT_FALSE(x.isType<AD::Const>());
  EXPECT_EQ(x.evaluateAt({x = TensorDouble({1, 2, 4})}).expression(),
            "Shape(3){1 2 4}");
  EXPECT_EQ(x.differentiate(x).expression(), "Shape(3, 3){1 0 0 0 1 0 0 0 1}");
  EXPECT_EQ(x.differentiate(y).expression(), "Shape(3, 3){0 0 0 0 0 0 0 0 0}");
  EXPECT_EQ(identifier(x), "x");

  EXPECT_EQ(y.expression(), "y");
  EXPECT_TRUE(y.isType<AD::Var>());
  EXPECT_FALSE(y.isType<AD::Const>());
  EXPECT_EQ(y.evaluateAt({x = TensorDouble({2})}).expression(), "y");
  EXPECT_EQ(y.differentiate(x).expression(), "Shape(3, 3){0 0 0 0 0 0 0 0 0}");
  EXPECT_EQ(y.differentiate(y).expression(), "Shape(3, 3){1 0 0 0 1 0 0 0 1}");

  EXPECT_EQ(z.expression(), "y");
  EXPECT_TRUE(z.isType<AD::Var>());
  EXPECT_FALSE(z.isType<AD::Const>());
  EXPECT_EQ(z.evaluateAt({x = TensorDouble({2})}).expression(), "y");
  EXPECT_EQ(z.differentiate(x).expression(), "Shape(3, 3){0 0 0 0 0 0 0 0 0}");
  EXPECT_EQ(z.differentiate(y).expression(), "Shape(3, 3){1 0 0 0 1 0 0 0 1}");
}

TEST(AD, ConstAndVar) {
  using NeuralNet::combineShapes;
  using TD = NeuralNet::Tensor<double>;
  using AD = AutomaticDifferentiation::AD<TD>;
  using NeuralNet::Shape;

  auto shape = Shape({3});
  auto shape2 = Shape({2});

  auto c = AD(TD({5, 2, 1}));
  auto x = AD("x", shape);
  auto y = AD("y", shape2);
  auto d = AD("d", TD({3, 2, 1}));

  EXPECT_EQ(value(D(c, x)), TD::sparse(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x, x)), TD::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(y, x)), TD::sparse(combineShapes(shape2, shape)));
  EXPECT_EQ(value(D(d, x)), TD::sparse(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(d, d)), TD::sparseEye(combineShapes(shape, shape)));

  EXPECT_EQ(value(D(c, {x})), TD::sparse(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x, {x})), TD::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x, {x, x})),
            TD::sparse(combineShapes(combineShapes(shape, shape), shape)));
  EXPECT_EQ(value(D(y, {x})), TD::sparse(combineShapes(shape2, shape)));
  EXPECT_EQ(value(D(d, {x})), TD::sparse(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(d, {d})), TD::sparseEye(combineShapes(shape, shape)));
}

TEST(AD, Exceptions) {
  using TD = NeuralNet::Tensor<double>;
  using AD = AutomaticDifferentiation::AD<TD>;
  using NeuralNet::Shape;

  auto shape = Shape({2});

  auto c = AD(TD({5.0, 2.0}));
  auto x = AD("x", shape);
  auto y = AD("y", shape);

  EXPECT_THROW(AD("1x", shape), std::invalid_argument);
  EXPECT_THROW(AD("", shape), std::invalid_argument);
  EXPECT_THROW(AD("abcdefghi", shape), std::invalid_argument);
  EXPECT_NO_THROW(AD("abcdefgh", shape));

  EXPECT_THROW(value(x), std::logic_error);
  EXPECT_THROW(identifier(c), std::logic_error);
  EXPECT_THROW(value(x + y), std::logic_error);
  EXPECT_THROW(identifier(x + y), std::logic_error);

  EXPECT_NO_THROW(value(c));
  EXPECT_NO_THROW(identifier(x));
}

TEST(AD, Op) {
  using T = NeuralNet::Tensor<double>;
  using AD = AutomaticDifferentiation::AD<T>;
  using NeuralNet::combineShapes;
  using NeuralNet::Shape;

  auto shape = Shape({3});

  auto x = AD("x", shape);
  auto y = AD("y", shape);
  auto z = AD("z", shape);

  EXPECT_EQ(value(D(x + y, y)), T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x + y, z)), T::sparse(combineShapes(shape, shape)));
  EXPECT_EQ((x + y).evaluateAt({x = T({1, 2, 2})}).expression(),
            "(Shape(3){1 2 2} + y)");

  EXPECT_EQ(value(D(T({1, 2, 1}) + y, y)),
            T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(y + T({1, 2, 1}), y)),
            T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(T({1, 2, 3}) - y, y)),
            -T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(y - T({3, 1, 2}), y)),
            T::sparseEye(combineShapes(shape, shape)));

  EXPECT_EQ(value(D(x + y, x)), T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x + y, y)), T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x - y, y)), -T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(y - x, y)), T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x + x - y, x)),
            2.0 * T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x + x - y, y)), -T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x - y - y + x, x)),
            2.0 * T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x - y - y + x, y)),
            -2.0 * T::sparseEye(combineShapes(shape, shape)));

  auto xx = reshape(x, Shape({1, 3}));
  EXPECT_EQ(value(xx.evaluateAt({x = T({1, 2, 3})})),
            T(T::Data2D({{1, 2, 3}})));
  EXPECT_EQ(
      value(multiply(x, {-1}, xx, {0, -1}).evaluateAt({x = T({1, 2, 3})})),
      T({14}));
  EXPECT_EQ(value(D(multiply(x, {-1}, xx, {0, -1}),
                    x).evaluateAt({x = T({1, 2, 3})})),
            T(T::Data2D({{2, 4, 6}})));
  EXPECT_EQ(value(D(3.0 * multiply(x, {-1}, xx, {0, -1}),
                    x).evaluateAt({x = T({1, 2, 3})})),
            T(T::Data2D({{6, 12, 18}})));

  /*
  EXPECT_EQ(value(D(3.0 * x * x + 5.0, x).evaluateAt({x = 2})), 12);

  EXPECT_EQ(value(D(x * y, x).evaluateAt({x = 3, y = 4})), 4);
  EXPECT_EQ(value(D(x * y, y).evaluateAt({x = 3, y = 4})), 3);

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
  */
}
/*
TEST(AD, Param) {
  using AD = AutomaticDifferentiation::AD<double>;

  auto x = AD("x");
  auto y = AD("y");
  auto c = AD("c", 5);

  auto expr = c * x;
  auto diff = D(expr, x);
  EXPECT_EQ(value(diff), 5);
  param(c) = 6;
  EXPECT_EQ(value(expr.evaluateAt({x = 2})), 12);
  EXPECT_EQ(value(diff), 6);

  auto diff2 = D(expr, c);
  EXPECT_EQ(value(diff2.evaluateAt({x = 1})), 1);
}
*/

int main(int argc, char** argv) {
  // Disables elapsed time by default.
  ::testing::GTEST_FLAG(print_time) = false;

  // This allows the user to override the flag on the command line.
  ::testing::InitGoogleTest(&argc, argv);

  google::InstallFailureSignalHandler();

  return RUN_ALL_TESTS();
}
