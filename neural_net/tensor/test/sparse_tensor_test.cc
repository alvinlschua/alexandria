#include <sstream>

#include "neural_net/tensor/tensor.h"
#include "neural_net/tensor/tensor_sparse.h"
#include "neural_net/tensor/tensor_dense.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

TEST(Tensor, Constructors) {
  using namespace NeuralNet;
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

  EXPECT_EQ(t1.size(), 3);

  auto t2 = Tensor<double>::outerProductEye(Shape({3}));
  EXPECT_EQ(t2.size(), 3);

  EXPECT_DOUBLE_EQ(t2.at({0,0}), 1.0);
  EXPECT_DOUBLE_EQ(t2.at({0,1}), 0.0);
  EXPECT_DOUBLE_EQ(t2.at({1,1}), 1.0);
  EXPECT_DOUBLE_EQ(t2.at({2,2}), 1.0);
  EXPECT_EQ(t2.size(), 3);

  auto t3 = Tensor<double>::outerProductZeros(Shape({3}));
  EXPECT_EQ(t3.size(), 0);
  
}

int main(int argc, char** argv) {
  // Disables elapsed time by default.
  ::testing::GTEST_FLAG(print_time) = false;

  // This allows the user to override the flag on the command line.
  ::testing::InitGoogleTest(&argc, argv);

  google::InstallFailureSignalHandler();

  return RUN_ALL_TESTS();
}
