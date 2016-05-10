#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

#include <sstream>

#include "tensor/tensor.h"

TEST(Tensor, Constructors) {
  using namespace Tensor;
  using namespace std;

  Tensor<> t1({1, 2, 3});

  EXPECT_DOUBLE_EQ(t1[{0}], 1.0);
  EXPECT_DOUBLE_EQ(t1[{1}], 2.0);
  EXPECT_DOUBLE_EQ(t1[{2}], 3.0);

  EXPECT_EQ(Shape({3}), t1.shape());

  // 3 x 2
  Tensor<> t2({{1, 2}, {3, 4}, {5, 6}});

  EXPECT_DOUBLE_EQ((t2[{0, 0}]), 1.0);
  EXPECT_DOUBLE_EQ((t2[{0, 1}]), 2.0);
  EXPECT_DOUBLE_EQ((t2[{1, 0}]), 3.0);
  EXPECT_DOUBLE_EQ((t2[{1, 1}]), 4.0);
  EXPECT_DOUBLE_EQ((t2[{2, 0}]), 5.0);
  EXPECT_DOUBLE_EQ((t2[{2, 1}]), 6.0);

  EXPECT_EQ(Shape({3, 2}), t2.shape());

  auto t3 = Tensor<>::fill(Shape({1, 2}), 3.0);
  for (const auto& item : t3) {
    EXPECT_DOUBLE_EQ(item, 3.0);
  }
  EXPECT_DOUBLE_EQ((t3[{0, 0}]), 3.0);
  EXPECT_DOUBLE_EQ((t3[{0, 1}]), 3.0);

  auto t4 = Tensor<>::zeros(Shape({2, 1}));
  for (const auto& item : t4) {
    EXPECT_DOUBLE_EQ(item, 0.0);
  }
  EXPECT_EQ(t4, Tensor<>::zeros(Shape({2, 1})));

  auto t5 = Tensor<>::ones(Shape({1, 3}));
  for (const auto& item : t5) {
    EXPECT_DOUBLE_EQ(item, 1.0);
  }
  EXPECT_EQ(t5, Tensor<>::ones(Shape({1, 3})));

  auto t6 =
      Tensor<>::generate(Shape({2, 3}), [](Address a) { return a[0] + a[1]; });
  EXPECT_DOUBLE_EQ((t6[{0, 0}]), 0.0);
  EXPECT_DOUBLE_EQ((t6[{0, 1}]), 1.0);
  EXPECT_DOUBLE_EQ((t6[{0, 2}]), 2.0);
  EXPECT_DOUBLE_EQ((t6[{1, 0}]), 1.0);
  EXPECT_DOUBLE_EQ((t6[{1, 1}]), 2.0);
  EXPECT_DOUBLE_EQ((t6[{1, 2}]), 3.0);
}

TEST(Tensor, AddSubtract) {
  using namespace Util;
  using namespace Tensor;

  auto t1 = Tensor<>({{1, 3, 2}, {4, 1, 4}});
  auto t2 = t1 + Tensor<>({{3, 1, 2}, {1, 4, 4}});
  t1 = plus(std::move(t1), Tensor<>({{3, 1, 2}, {1, 4, 4}}));
  EXPECT_EQ(t2, Tensor<>({{4, 4, 4}, {5, 5, 8}}));
  EXPECT_EQ(t2, t1);

  auto t3 = Tensor<>({{1, 3, 2}, {4, 1, 4}});
  t3 = plus(std::move(t3), Tensor<>({{3, 1, 2}, {1, 4, 4}}));
  EXPECT_EQ(t3, t2);

  t1 = Tensor<>({{1, 3, 2}, {4, 1, 4}});
  t2 = t1 - Tensor<>({{3, 1, 2}, {1, 4, 4}});
  t1 = minus(std::move(t1), Tensor<>({{3, 1, 2}, {1, 4, 4}}));
  /*
  t1.subtract(Tensor<>::make(Shape({2, 3}), {3, 1, 3, 1, 2, 1}));
  EXPECT_EQ(t1, Tensor<>::make(Shape({2, 3}), {1, 3, 1, 4, 3, 7}));

  t1 = -t1;
  t1.negate();
  EXPECT_EQ(t1, Tensor<>::make(Shape({2, 3}), {-1, -3, -1, -4, -3, -7}));
  */
}

TEST(Tensor, Multiply) {
  using namespace Util;
  using namespace Tensor;
  using namespace std;

  auto t1 = Tensor<>({{1, 3, 2}, {4, 1, 4}});    // 2 x 3
  auto t2 = Tensor<>({{1, 3}, {2, 4}, {1, 4}});  // 3 x 2

  // Tensor 2 x 3 || 3 x 2
  auto t3 = multiply(t1, {0, -1}, t2, {-1, 1});
  EXPECT_EQ(t3, Tensor<>({{9, 23}, {10, 32}}));

  auto t4 = multiply(t1, {0, 1}, t1, {0, 1});
  EXPECT_EQ(t4, Tensor<>({{1, 9, 4}, {16, 1, 16}}));

  auto t5 = multiply(t1, {0, 1}, Tensor<>({1, 3, 2}), {1});
  EXPECT_EQ(t5, Tensor<>({{1, 9, 4}, {4, 3, 8}}));

  // 2 3 2 x 2 3
  auto t6 = Tensor<>({{{1, 3}, {2, 4}, {1, 4}}, {{1, 2}, {3, 2}, {4, 3}}});
  t6 = multiply(t6, {0, -1, -2}, Tensor<>({{1, 3, 2}, {4, 1, 4}}), {-2, -1});
  EXPECT_EQ(t6, Tensor<>({41, 40}));

  auto t7 = Tensor<>({{1, 3, 2}, {4, 1, 4}});  // 2 x 3
  t7 = multiply(t7, {0, 1}, Tensor<>({2, 4}), {0});
  EXPECT_EQ(t7, Tensor<>({{2, 6, 4}, {16, 4, 16}}));

  auto t8 = multiply(Tensor<>({1, 3, 2}), {0}, Tensor<>({2, 1, 3}), {1});
  EXPECT_EQ(t8, Tensor<>({{2, 1, 3}, {6, 3, 9}, {4, 2, 6}}));
}

/*
TEST(Tensor, timeMultiply) {
  using namespace Sonnet::Util;
  using namespace Sonnet::Tensor;
  using namespace std;

  auto x = Tensor<>::random(Shape({1000, 100}));
  auto y = Tensor<>::random(Shape({100, 1000}));
  x.multiply(y);
}

TEST(Tensor, mask) {
  using namespace Sonnet::Util;
  using namespace Sonnet::Tensor;
  using namespace std;

  auto t1 =
      Tensor<>::make(Shape({2, 3, 2}), {1, 3, 2, 4, 1, 4, 2, 3, 2, 3, 4, 3});

  auto t2 = t1;
  t2.fold({0});
  EXPECT_EQ(t2, Tensor<>::make(Shape({3, 2}), {3, 6, 4, 7, 5, 7}));

  auto t3 = t1;
  t3.fold({1, 2});
  EXPECT_EQ(t3, Tensor<>::make(Shape({2}), {15, 17}));

  auto t4 = t1;
  t4.maskIf(t4, [](double value) { return value <= 2; });
  EXPECT_EQ(t4, Tensor<>::make(Shape({2, 3, 2}),
                               {0, 3, 0, 4, 0, 4, 0, 3, 0, 3, 4, 3}));

  auto t5 = t1;
  //  t5.maskByIndex([](const DimIndex& index) { return index[0] == 0; });
  t5.maskElementIndices(
      {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {0, 2, 0}, {0, 2, 1}});
  EXPECT_EQ(t5, Tensor<>::make(Shape({2, 3, 2}),
                               {0, 0, 0, 0, 0, 0, 2, 3, 2, 3, 4, 3}));
}

TEST(Tensor, applySlice) {
  using namespace Sonnet::Util;
  using namespace Sonnet::Tensor;
  using namespace std;

  auto t1 =
      Tensor<>::make(Shape({2, 3, 2}), {1, 3, 2, 4, 1, 4, 2, 3, 2, 3, 4, 3});
  auto t2 = t1;

  t2.rotate(0, 1);
  EXPECT_EQ(t2, Tensor<>::make(Shape({2, 3, 2}),
                               {2, 3, 2, 3, 4, 3, 1, 3, 2, 4, 1, 4}));

  auto t3 = t1;
  t3.accumulate(0);
  EXPECT_EQ(t3, Tensor<>::make(Shape({2, 3, 2}),
                               {1, 3, 2, 4, 1, 4, 3, 6, 4, 7, 5, 7}));

  auto t4 = t1;
  t4.reverseAccumulate(0);
  EXPECT_EQ(t4, Tensor<>::make(Shape({2, 3, 2}),
                               {3, 6, 4, 7, 5, 7, 2, 3, 2, 3, 4, 3}));

  auto t5 = t1;
  t5.reverse(2);
  EXPECT_EQ(t5, Tensor<>::make(Shape({2, 3, 2}),
                               {3, 1, 4, 2, 4, 1, 3, 2, 3, 2, 3, 4}));
}

TEST(Tensor, nonChainedCommands) {
  using namespace Sonnet::Util;
  using namespace Sonnet::Tensor;
  using namespace std;

  auto t1 = Tensor<>::make(Shape({3}), {2, 3, 2});
  EXPECT_DOUBLE_EQ(t1.innerProduct(t1), 17.0);
  EXPECT_DOUBLE_EQ(t1.product(), 12.0);
  EXPECT_DOUBLE_EQ(t1.sum(), 7.0);
}

TEST(Tensor, covolution) {
  using namespace Sonnet::Util;
  using namespace Sonnet::Tensor;
  using namespace std;

  auto t = Tensor<>::make(Shape({5}), {2, 3, 2, 1, 2});
  auto kernel = Tensor<>::make(Shape({3}), {1, 1, 1});

  auto t1 = t;
  t1.convolve1D(kernel);
  CHECK(t1 == Tensor<>::make(Shape({3}), {7, 6, 5}));

  auto t2 = t;
  t2.convolve1DExtended(kernel);
  CHECK(t2 == Tensor<>::make(Shape({7}), {2, 5, 7, 6, 5, 3, 2}));

  auto t3 = t;
  t3.convolve1DExtended(kernel, 2);
  CHECK(t3 == Tensor<>::make(Shape({9}), {2, 3, 4, 4, 6, 4, 4, 1, 2}));

  auto t4 = t;
  t4.convolve1D(kernel, 2);
  CHECK(t4 == Tensor<>::make(Shape({1}), {6}));
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
