#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

#include "tensor/shape.h"
#include "tensor/helpers.h"

TEST(Helpers, indicesUnique) {
  using namespace Tensor;
  using namespace std;

  Indices indices1({1, 2, 3, 4});
  EXPECT_TRUE(indicesUnique(indices1));

  Indices indices2({1, 2, 3, 2});
  EXPECT_FALSE(indicesUnique(indices2));
}

TEST(Helpers, multiplyShapes) {
  using namespace Tensor;

  Shape shape1({3, 4, 2});
  Shape shape2({3, 4, 3, 2});

  Shape result_shape;
  Shape common_shape;

  std::tie(result_shape, common_shape) =
      multiplyShapes(shape1, {2, 0, 1}, shape2, {3, 4, 2, 5});

  EXPECT_EQ(result_shape, Shape({4, 2, 3, 3, 4, 2}));
  EXPECT_EQ(common_shape, Shape());

  std::tie(result_shape, common_shape) =
      multiplyShapes(shape1, {2, 0, 1}, shape2, {3, 4, 2, 1});
  EXPECT_EQ(result_shape, Shape({4, 2, 3, 3, 4}));
  EXPECT_EQ(common_shape, Shape());

  std::tie(result_shape, common_shape) =
      multiplyShapes(shape1, {2, 0, 1}, shape2, {3, 4, 5, 1});
  EXPECT_EQ(result_shape, Shape({4, 2, 3, 3, 4, 3}));
  EXPECT_EQ(common_shape, Shape());

  std::tie(result_shape, common_shape) =
      multiplyShapes(shape1, {2, 0, 1}, shape2, {2, 0, 3, 1});
  EXPECT_EQ(result_shape, Shape({4, 2, 3, 3}));
  EXPECT_EQ(common_shape, Shape());

  Shape result_shape2;
  std::tie(result_shape2, common_shape) =
      multiplyShapes(shape2, {2, 0, 3, 1}, shape1, {2, 0, 1});
  EXPECT_EQ(result_shape, result_shape2);

  std::tie(result_shape, common_shape) =
      multiplyShapes(shape1, {-1, 0, -2}, shape2, {-1, 0, 1, -2});
  EXPECT_EQ(result_shape, Shape({4, 3}));
  EXPECT_EQ(common_shape, Shape({3, 2}));

  std::tie(result_shape, common_shape) =
      multiplyShapes(shape1, {-1, 0, 2}, shape2, {-1, 0, 1, 2});
  EXPECT_EQ(result_shape, Shape({4, 3, 2}));
  EXPECT_EQ(common_shape, Shape({3}));

  EXPECT_THROW(multiplyShapes(shape1, {0, 1}, shape2, {2, 0, 3, 1}),
               std::invalid_argument);
  EXPECT_THROW(multiplyShapes(shape1, {2, 0, 1}, shape2, {0, 3, 1}),
               std::invalid_argument);
  EXPECT_THROW(multiplyShapes(shape1, {2, 0, 1}, shape2, {3, 4, 5, 2}),
               std::invalid_argument);
  EXPECT_THROW(multiplyShapes(shape1, {2, 1, 1}, shape2, {3, 4, 5, 1}),
               std::invalid_argument);
  EXPECT_THROW(multiplyShapes(shape1, {2, 0, 1}, shape2, {3, 4, 1, 1}),
               std::invalid_argument);
  EXPECT_THROW(multiplyShapes(shape1, {-1, -1, 2}, shape2, {-1, 0, 1, 2}),
               std::invalid_argument);
  EXPECT_THROW(multiplyShapes(shape1, {-1, 0, 2}, shape2, {-1, 0, 1, -2}),
               std::invalid_argument);
  EXPECT_THROW(multiplyShapes(shape1, {-2, 0, 2}, shape2, {-1, 0, 1, -2}),
               std::invalid_argument);
  EXPECT_THROW(multiplyShapes(shape1, {-2, 0, 2}, shape2, {-2, 0, 1, 2}),
               std::invalid_argument);
}

int main(int argc, char** argv) {
  // Disables elapsed time by default.
  ::testing::GTEST_FLAG(print_time) = false;

  // This allows the user to override the flag on the command line.
  ::testing::InitGoogleTest(&argc, argv);

  google::InstallFailureSignalHandler();

  return RUN_ALL_TESTS();
}
