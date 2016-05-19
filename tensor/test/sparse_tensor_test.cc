#include <sstream>

#include "tensor/tensor.h"
#include "tensor/tensor_dense.h"
#include "tensor/tensor_sparse.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

TEST(Tensor, Constructors) {
  using namespace Alexandria;
  using namespace std;

  auto t1 = Tensor<double>::sparse(Shape({3}));

  EXPECT_EQ(t1.size(), 0);
  EXPECT_DOUBLE_EQ(t1.at({0}), 0.0);
  EXPECT_DOUBLE_EQ(t1.at({1}), 0.0);
  EXPECT_DOUBLE_EQ(t1.at({2}), 0.0);

  // Accessing a non constant sparse tensor with operator [] will change the
  // size. Use at instead.
  EXPECT_EQ(t1.size(), 0);
  EXPECT_DOUBLE_EQ(t1[{0}], 0.0);
  EXPECT_DOUBLE_EQ(t1[{1}], 0.0);
  EXPECT_DOUBLE_EQ(t1[{2}], 0.0);

  EXPECT_EQ(t1.size(), 0);

  auto t2 = Tensor<double>::sparseEye(Shape({3, 3}));
  EXPECT_EQ(t2.size(), 3);

  EXPECT_DOUBLE_EQ(t2.at({0, 0}), 1.0);
  EXPECT_DOUBLE_EQ(t2.at({0, 1}), 0.0);
  EXPECT_DOUBLE_EQ(t2.at({1, 1}), 1.0);
  EXPECT_DOUBLE_EQ(t2.at({2, 2}), 1.0);
  EXPECT_EQ(t2.size(), 3);

  auto t3 = Tensor<double>::sparse(Shape({3, 3}));
  EXPECT_EQ(t3.size(), 0);

  auto t4 = Tensor<double>::sparseEye(Shape({3, 3}));
  auto t5 = Tensor<double>::zeros(Shape({3, 3}));

  t5.set({0, 0}, 1);
  t5.set({1, 1}, 1);
  EXPECT_NE(t4, t5);

  t5.set({2, 2}, 1);
  EXPECT_EQ(t4, t5);

  auto t6 = Tensor<double>::sparseEye(Shape({3, 3}));
  EXPECT_EQ(t4, t6);

  auto t7 = Tensor<double>::sparse(Shape({3, 3}));
  EXPECT_NE(t4, t7);

  auto t8 = Tensor<double>::sparseEye(Shape({3, 1, 3, 1}));
  EXPECT_NE(t4, t8);
}

TEST(Tensor, Op) {
  using namespace Alexandria;
  using namespace std;

  auto t1 = Tensor<double>::sparse(Shape({3}));
  t1.set({0}, 4);
  auto t2 = Tensor<double>::sparse(Shape({3}));
  t2.set({1}, 5);

  EXPECT_EQ(t1 + t2, Tensor<double>({4, 5, 0}));
  EXPECT_EQ(t2 + t1, Tensor<double>({4, 5, 0}));
  EXPECT_EQ(t1 - t2, Tensor<double>({4, -5, 0}));
  EXPECT_EQ(t2 - t1, Tensor<double>({-4, 5, 0}));
  EXPECT_EQ(t1 - t1, Tensor<double>({0, 0, 0}));
  EXPECT_EQ(t2 - t2, Tensor<double>({0, 0, 0}));

  EXPECT_EQ(multiply(Tensor<double>({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}), {-1, 0},
                     Tensor<double>({1, 2, 3}), {-1}),
            Tensor<double>({1, 2, 3}));
  EXPECT_EQ(multiply(Tensor<double>({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}), {0, -1},
                     Tensor<double>({1, 2, 3}), {-1}),
            Tensor<double>({1, 2, 3}));
  EXPECT_EQ(multiply(Tensor<double>::sparseEye(Shape({3, 3})), {-1, 0},
                     Tensor<double>({1, 2, 3}), {-1}),
            Tensor<double>({1, 2, 3}));
}

TEST(Tensor, Serialize) {
  using namespace Alexandria;
  using namespace std;

  auto t1 = Tensor<double>::sparse(Shape({3}));
  t1.set({0}, 4);
  auto t2 = Tensor<double>::sparse(Shape({3}));
  t2.set({1}, 5);

  ostringstream sout;
  ArchiveOut ar_out(&sout);

  ar_out % t1;
  ar_out % t2;

  Tensor<double> t3;
  Tensor<double> t4;

  istringstream sin(sout.str());
  ArchiveIn ar_in(&sin);

  ar_in % t3;
  EXPECT_EQ(t1, t3);

  ar_in % t4;
  EXPECT_EQ(t2, t4);
}

int main(int argc, char** argv) {
  // Disables elapsed time by default.
  ::testing::GTEST_FLAG(print_time) = false;

  // This allows the user to override the flag on the command line.
  ::testing::InitGoogleTest(&argc, argv);

  google::InstallFailureSignalHandler();

  return RUN_ALL_TESTS();
}
