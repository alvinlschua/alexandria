#include <cmath>

#include "automatic_differentiation/ad_tensor.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

TEST(AD, Basic) {
  using T = Alexandria::Tensor<double>;
  using AD = Alexandria::AD<T>;
  using Alexandria::Shape;

  AD c(T({5.0, 2.0, 3.0}));

  AD x("x", Shape({3}));
  auto y = AD("y", Shape({3}));
  auto z = y;

  EXPECT_EQ(c.expression(), "Shape(3){ 5 2 3 }");
  EXPECT_TRUE(c.isType<AD::Const>());
  EXPECT_FALSE(c.isType<AD::Var>());
  EXPECT_EQ(c.evaluateAt({x = T({2, 2, 3})}).expression(), "Shape(3){ 5 2 3 }");
  EXPECT_EQ(c.differentiate(x).expression(),
            "Shape(3, 3){ 0 0 0 0 0 0 0 0 0 }");
  EXPECT_TRUE(c.differentiate(x).isType<AD::Const>());
  EXPECT_EQ(value(c), T({5, 2, 3}));

  EXPECT_EQ(x.expression(), "x");
  EXPECT_TRUE(x.isType<AD::Var>());
  EXPECT_FALSE(x.isType<AD::Const>());
  EXPECT_EQ(x.evaluateAt({x = T({1, 2, 4})}).expression(), "Shape(3){ 1 2 4 }");
  EXPECT_EQ(x.differentiate(x).expression(),
            "Shape(3, 3){ 1 0 0 0 1 0 0 0 1 }");
  EXPECT_EQ(x.differentiate(y).expression(),
            "Shape(3, 3){ 0 0 0 0 0 0 0 0 0 }");
  EXPECT_EQ(identifier(x), "x");

  EXPECT_EQ(y.expression(), "y");
  EXPECT_TRUE(y.isType<AD::Var>());
  EXPECT_FALSE(y.isType<AD::Const>());
  EXPECT_EQ(y.evaluateAt({x = T({2})}).expression(), "y");
  EXPECT_EQ(y.differentiate(x).expression(),
            "Shape(3, 3){ 0 0 0 0 0 0 0 0 0 }");
  EXPECT_EQ(y.differentiate(y).expression(),
            "Shape(3, 3){ 1 0 0 0 1 0 0 0 1 }");

  EXPECT_EQ(z.expression(), "y");
  EXPECT_TRUE(z.isType<AD::Var>());
  EXPECT_FALSE(z.isType<AD::Const>());
  EXPECT_EQ(z.evaluateAt({x = T({2})}).expression(), "y");
  EXPECT_EQ(z.differentiate(x).expression(),
            "Shape(3, 3){ 0 0 0 0 0 0 0 0 0 }");
  EXPECT_EQ(z.differentiate(y).expression(),
            "Shape(3, 3){ 1 0 0 0 1 0 0 0 1 }");
}

TEST(AD, ConstAndVar) {
  using T = Alexandria::Tensor<double>;
  using AD = Alexandria::AD<T>;
  using Alexandria::Shape;

  auto shape = Shape({2});

  auto c = AD(T({5.0, 2.0}));
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

TEST(AD, UnaryOp) {
  using T = Alexandria::Tensor<double>;
  using AD = Alexandria::AD<T>;
  using Alexandria::Shape;

  auto shape = Shape({3});
  auto x = AD("x", shape);
  auto eye = T::sparseEye(combineShapes(shape, shape));

  auto xx = reshape(x, Shape({1, 3}));
  EXPECT_EQ(value(xx.evaluateAt({x = T({1, 2, 3})})),
            T(T::Data2d({{1, 2, 3}})));
  EXPECT_EQ(
      value(multiply(x, {-1}, xx, {0, -1}).evaluateAt({x = T({1, 2, 3})})),
      T({14}));
  EXPECT_EQ(value(D(multiply(x, {-1}, xx, {0, -1}),
                    x).evaluateAt({x = T({1, 2, 3})})),
            T(T::Data2d({{2, 4, 6}})));
  EXPECT_EQ(value(D(3.0 * multiply(x, {-1}, xx, {0, -1}),
                    x).evaluateAt({x = T({1, 2, 3})})),
            T(T::Data2d({{6, 12, 18}})));

  EXPECT_EQ(value(sigmoid(x).evaluateAt({x = T({-1, 0, 1})})),
            T({1.0 / (1.0 + exp(1)), 1.0 / 2.0, 1.0 / (1.0 + exp(-1))}));
  std::cout << D(sigmoid(x), x).evaluateAt({x = T({-1, 0, 1})}).expression() << std::endl;
  EXPECT_EQ(value(D(sigmoid(x), x).evaluateAt({x = T({-1, 0, 1})})),
            T({{1.0 / (1.0 + exp(1)) * (1.0 - 1.0 / (1.0 + exp(1))), 0, 0},
               {0, 1.0 / 4.0, 0},
               {0, 0, 1.0 / (1.0 + exp(-1)) * (1.0 - 1.0 / (1.0 + exp(-1)))}}));
  
  EXPECT_EQ(value(log(x).evaluateAt({x = T({0.1, 1, 2})})),
            T({log(0.1), log(1), log(2)}));
  EXPECT_EQ(value(D(log(x), x).evaluateAt({x = T({0.1, 1, 2})})),
            T({{1 / 0.1, 0, 0}, {0, 1.0, 0}, {0, 0, 0.5}}));

  EXPECT_EQ(value(reciprocal(x).evaluateAt({x = T({0.1, 1, 2})})),
            T({1.0 / 0.1, 1, 0.5}));

  EXPECT_EQ(value(D(reciprocal(x), x).evaluateAt({x = T({0.1, 1, 2})})),
            T({{-1.0 / 0.01, 0, 0}, {0, -1.0, 0}, {0, 0, -0.25}}));
}

TEST(AD, Op) {
  using T = Alexandria::Tensor<double>;
  using AD = Alexandria::AD<T>;
  using Alexandria::Shape;

  auto shape = Shape({3});

  auto x = AD("x", shape);
  auto y = AD("y", shape);
  auto z = AD("z", shape);

  EXPECT_EQ(value(D(x + y, y)), T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x + y, z)), T::sparse(combineShapes(shape, shape)));
  EXPECT_EQ((x + y).evaluateAt({x = T({1, 2, 2})}).expression(),
            "(Shape(3){ 1 2 2 } + y)");

  EXPECT_EQ(value(D(T({1, 2, 1}) + y, y)),
            T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(y + T({1, 2, 1}), y)),
            T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(T({1, 2, 3}) - y, y)),
            T::constDiagonal(combineShapes(shape, shape), -1));
  EXPECT_EQ(value(D(y - T({3, 1, 2}), y)),
            T::sparseEye(combineShapes(shape, shape)));

  EXPECT_EQ(value(D(x + y, x)), T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x + y, y)), T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x - y, y)),
            T::constDiagonal(combineShapes(shape, shape), -1));
  EXPECT_EQ(value(D(y - x, y)), T::sparseEye(combineShapes(shape, shape)));
  EXPECT_EQ(value(D(x + x - y, x)),
            T::constDiagonal(combineShapes(shape, shape), 2.0));
  EXPECT_EQ(value(D(x + x - y, y)),
            T::constDiagonal(combineShapes(shape, shape), -1));
  EXPECT_EQ(value(D(x - y - y + x, x)),
            T::constDiagonal(combineShapes(shape, shape), 2.0));
  EXPECT_EQ(value(D(x - y - y + x, y)),
            T::constDiagonal(combineShapes(shape, shape), -2.0));
  EXPECT_EQ(value(D(x + y / 2.0, y)),
            T::constDiagonal(combineShapes(shape, shape), 1.0 / 2.0));
}

TEST(AD, Param) {
  using T = Alexandria::Tensor<double>;
  using AD = Alexandria::AD<T>;
  using Alexandria::Shape;

  auto shape = Shape({3});

  auto t = T({{1, 3, 1}, {1, 4, 2}, {2, 1, 3}});
  auto t2 = T({{1, 3, 3}, {2, 4, 2}, {2, 1, 3}});
  auto x = AD("x", shape);
  auto y = AD("y", shape);
  auto c = AD("c", t);

  auto expr = multiply(c, {0, -1}, x, {-1});
  auto diff = D(expr, x);
  auto diff2 = D(expr, c);

  EXPECT_EQ(value(diff), t);
  param(c) = t2;

  EXPECT_EQ(value(expr.evaluateAt({x = T({1, 1, 1})})), T({7, 8, 6}));
  EXPECT_EQ(value(diff), t2);
  EXPECT_EQ(value(diff2.evaluateAt({x = T({1, 2, 4})})),
            multiply(T({1, 2, 4}), {2}, T::sparseEye(Shape({3, 3})), {0, 1}));
}

int main(int argc, char** argv) {
  // Disables elapsed time by default.
  ::testing::GTEST_FLAG(print_time) = false;

  // This allows the user to override the flag on the command line.
  ::testing::InitGoogleTest(&argc, argv);

  google::InstallFailureSignalHandler();

  return RUN_ALL_TESTS();
}
