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
#include "tensor/tensor_base.h"
#include "tensor/tensor_dense.h"
#include "tensor/tensor_sparse.h"

TEST(Tensor, Constructors) {
  using namespace Alexandria;
  using namespace std;

  Tensor<double> t1({1.0, 2.0, 3.0});

  EXPECT_DOUBLE_EQ(t1[{0}], 1.0);
  EXPECT_DOUBLE_EQ(t1[{1}], 2.0);
  EXPECT_DOUBLE_EQ(t1[{2}], 3.0);

  EXPECT_EQ(Shape({3}), t1.shape());

  // 3 x 2
  Tensor<double> t2({{1, 2}, {3, 4}, {5, 6}});

  EXPECT_DOUBLE_EQ((t2[{0, 0}]), 1.0);
  EXPECT_DOUBLE_EQ((t2[{0, 1}]), 2.0);
  EXPECT_DOUBLE_EQ((t2[{1, 0}]), 3.0);
  EXPECT_DOUBLE_EQ((t2[{1, 1}]), 4.0);
  EXPECT_DOUBLE_EQ((t2[{2, 0}]), 5.0);
  EXPECT_DOUBLE_EQ((t2[{2, 1}]), 6.0);

  EXPECT_EQ(Shape({3, 2}), t2.shape());

  auto t3 = Tensor<double>::fill(Shape({1, 2}), 3.0);
  EXPECT_DOUBLE_EQ((t3[{0, 0}]), 3.0);
  EXPECT_DOUBLE_EQ((t3[{0, 1}]), 3.0);

  auto t4 = Tensor<double>::zeros(Shape({2, 1}));
  EXPECT_EQ(t4, Tensor<double>::zeros(Shape({2, 1})));

  auto t5 = Tensor<double>::ones(Shape({1, 3}));
  EXPECT_EQ(t5, Tensor<double>::ones(Shape({1, 3})));

  auto t6 = Tensor<double>::generate(Shape({2, 3}),
                                     [](Address a) { return a[0] + a[1]; });
  EXPECT_DOUBLE_EQ((t6[{0, 0}]), 0.0);
  EXPECT_DOUBLE_EQ((t6[{0, 1}]), 1.0);
  EXPECT_DOUBLE_EQ((t6[{0, 2}]), 2.0);
  EXPECT_DOUBLE_EQ((t6[{1, 0}]), 1.0);
  EXPECT_DOUBLE_EQ((t6[{1, 1}]), 2.0);
  EXPECT_DOUBLE_EQ((t6[{1, 2}]), 3.0);

  Tensor<double> t7(Tensor<double>::Data2D({{1.0, 2.0, 3.0}}));
  EXPECT_DOUBLE_EQ((t7[{0, 0}]), 1.0);
  EXPECT_DOUBLE_EQ((t7[{0, 1}]), 2.0);
  EXPECT_DOUBLE_EQ((t7[{0, 2}]), 3.0);
}

TEST(Tensor, AddSubtract) {
  using namespace Alexandria;

  auto t1 = Tensor<double>({{1, 3, 2}, {4, 1, 4}});
  auto t2 = t1 + Tensor<double>({{3, 1, 2}, {1, 4, 4}});
  EXPECT_EQ(t2, Tensor<double>({{4, 4, 4}, {5, 5, 8}}));
  EXPECT_EQ(t1 + t1, Tensor<double>({{2, 6, 4}, {8, 2, 8}}));

  auto t3 = Tensor<double>({{1, 3, 2}, {4, 1, 4}});
  t3 = plus(std::move(t3), Tensor<double>({{3, 1, 2}, {1, 4, 4}}));
  EXPECT_EQ(t3, t2);

  t1 = Tensor<double>({{1, 3, 2}, {4, 1, 4}});
  t2 = t1 - Tensor<double>({{3, 1, 2}, {1, 4, 4}});
  EXPECT_EQ(t2, Tensor<double>({{-2, 2, 0}, {3, -3, 0}}));

  t1 = minus(std::move(t1), Tensor<double>({{3, 1, 2}, {1, 4, 4}}));
  EXPECT_EQ(t1, t2);

  t1 = Tensor<double>({{1, 3, 1}, {4, 3, 7}});
  auto t4 = -t1;
  EXPECT_EQ(t4, Tensor<double>({{-1, -3, -1}, {-4, -3, -7}}));
  EXPECT_EQ(t1, Tensor<double>({{1, 3, 1}, {4, 3, 7}}));

  auto t5 = 3.0 * t1;
  auto t6 = t1 * 3.0;
  EXPECT_EQ(t5, t6);
  EXPECT_EQ(t5, Tensor<double>({{3, 9, 3}, {12, 9, 21}}));

  auto t7 = t1 / 2.0;
  EXPECT_EQ(t7, Tensor<double>({{1.0 / 2, 3.0 / 2, 1.0 / 2},
                                {4.0 / 2, 3.0 / 2, 7.0 / 2}}));
}

TEST(Tensor, Multiply) {
  using namespace Alexandria;
  using namespace std;

  auto t1 = Tensor<double>({{1, 3, 2}, {4, 1, 4}});    // 2 x 3
  auto t2 = Tensor<double>({{1, 3}, {2, 4}, {1, 4}});  // 3 x 2

  // Tensor 2 x 3 || 3 x 2
  auto t3 = multiply<double>(t1, {0, -1}, t2, {-1, 1});
  EXPECT_EQ(t3, Tensor<double>({{9, 23}, {10, 32}}));

  auto t4 = multiply<double>(t1, {0, 1}, t1, {0, 1});
  EXPECT_EQ(t4, Tensor<double>({{1, 9, 4}, {16, 1, 16}}));

  auto t5 = multiply<double>(t1, {0, 1}, Tensor<double>({1, 3, 2}), {1});
  EXPECT_EQ(t5, Tensor<double>({{1, 9, 4}, {4, 3, 8}}));

  // (2 3 2) x (2 3)
  auto t6 =
      Tensor<double>({{{1, 3}, {2, 4}, {1, 4}}, {{1, 2}, {3, 2}, {4, 3}}});
  t6 = multiply<double>(t6, {0, -1, -2}, Tensor<double>({{1, 3, 2}, {4, 1, 4}}),
                        {-2, -1});
  EXPECT_EQ(t6, Tensor<double>({41, 40}));

  auto t7 = Tensor<double>({{1, 3, 2}, {4, 1, 4}});  // 2 x 3
  t7 = multiply<double>(t7, {0, 1}, Tensor<double>({2, 4}), {0});
  EXPECT_EQ(t7, Tensor<double>({{2, 6, 4}, {16, 4, 16}}));

  auto t8 = multiply<double>(Tensor<double>({1, 3, 2}), {0},
                             Tensor<double>({2, 1, 3}), {1});
  EXPECT_EQ(t8, Tensor<double>({{2, 1, 3}, {6, 3, 9}, {4, 2, 6}}));

  auto t9 = multiply<double>(
      Tensor<double>({1, 3, 2}), {-1},
      Tensor<double>(Tensor<double>::Data2D({{2, 1, 3}})), {0, -1});
  EXPECT_EQ(t9, Tensor<double>({11}));
}

TEST(Tensor, Serialize) {
  using namespace Alexandria;
  using namespace std;

  auto t1 = Tensor<double>({{1, 3, 2}, {4, 1, 4}});    // 2 x 3
  auto t2 = Tensor<double>({{1, 3}, {2, 4}, {1, 4}});  // 3 x 2

  ostringstream sout;
  ArchiveOut ar_out(&sout);

  ar_out % t1;
  ar_out % t2;

  std::vector<Tensor<double>> vec;
  vec.emplace_back(Tensor<double>({1, 2, 3, 2, 1}));
  vec.emplace_back(Tensor<double>({{1, 2}, {3, 4}, {1, 2}}));

  ar_out % vec;

  istringstream sin(sout.str());
  ArchiveIn ar_in(&sin);

  Tensor<double> t3;
  Tensor<double> t4;

  ar_in % t3;
  EXPECT_EQ(t1, t3);

  ar_in % t4;
  EXPECT_EQ(t2, t4);

  std::vector<Tensor<double>> vec2;
  ar_in % vec2;

  EXPECT_EQ(vec.size(), vec2.size());
  EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), vec2.cbegin()));
}

int main(int argc, char** argv) {
  // Disables elapsed time by default.
  ::testing::GTEST_FLAG(print_time) = false;

  // This allows the user to override the flag on the command line.
  ::testing::InitGoogleTest(&argc, argv);

  google::InstallFailureSignalHandler();

  return RUN_ALL_TESTS();
}
