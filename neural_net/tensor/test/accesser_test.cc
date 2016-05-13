#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshift-sign-overflow"
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#include "gtest/gtest.h"
#pragma clang diagnostic pop

#include <sstream>

#include "neural_net/tensor/accesser.h"

TEST(Accesser, Basic1) {
  using namespace Util;
  using namespace NeuralNet;
  using namespace std;

  Shape shape({2, 1, 4});
  Accesser accesser(&shape);

  EXPECT_EQ(accesser.flatIndex({0, 0, 0}), 0ul);
  EXPECT_EQ(accesser.flatIndex({0, 0, 1}), 1ul);
  EXPECT_EQ(accesser.flatIndex({0, 0, 2}), 2ul);
  EXPECT_EQ(accesser.flatIndex({0, 0, 3}), 3ul);
  EXPECT_EQ(accesser.flatIndex({1, 0, 0}), 4ul);
  EXPECT_EQ(accesser.flatIndex({1, 0, 1}), 5ul);
  EXPECT_EQ(accesser.flatIndex({1, 0, 2}), 6ul);
  EXPECT_EQ(accesser.flatIndex({1, 0, 3}), 7ul);

  EXPECT_EQ(accesser.address(0), Address({0, 0, 0}));
  EXPECT_EQ(accesser.address(1), Address({0, 0, 1}));
  EXPECT_EQ(accesser.address(2), Address({0, 0, 2}));
  EXPECT_EQ(accesser.address(3), Address({0, 0, 3}));
  EXPECT_EQ(accesser.address(4), Address({1, 0, 0}));
  EXPECT_EQ(accesser.address(5), Address({1, 0, 1}));
  EXPECT_EQ(accesser.address(6), Address({1, 0, 2}));
  EXPECT_EQ(accesser.address(7), Address({1, 0, 3}));

  auto shape2 = Shape({1, 2, 4});
  auto accesser2 = Accesser(&shape2);
  EXPECT_EQ(accesser2.flatIndex({0, 0, 0}), 0ul);
  EXPECT_EQ(accesser2.flatIndex({0, 0, 1}), 1ul);
  EXPECT_EQ(accesser2.flatIndex({0, 0, 2}), 2ul);
  EXPECT_EQ(accesser2.flatIndex({0, 0, 3}), 3ul);
  EXPECT_EQ(accesser2.flatIndex({0, 1, 0}), 4ul);
  EXPECT_EQ(accesser2.flatIndex({0, 1, 1}), 5ul);
  EXPECT_EQ(accesser2.flatIndex({0, 1, 2}), 6ul);
  EXPECT_EQ(accesser2.flatIndex({0, 1, 3}), 7ul);

  EXPECT_EQ(accesser2.address(0), Address({0, 0, 0}));
  EXPECT_EQ(accesser2.address(1), Address({0, 0, 1}));
  EXPECT_EQ(accesser2.address(2), Address({0, 0, 2}));
  EXPECT_EQ(accesser2.address(3), Address({0, 0, 3}));
  EXPECT_EQ(accesser2.address(4), Address({0, 1, 0}));
  EXPECT_EQ(accesser2.address(5), Address({0, 1, 1}));
  EXPECT_EQ(accesser2.address(6), Address({0, 1, 2}));
  EXPECT_EQ(accesser2.address(7), Address({0, 1, 3}));
}

TEST(Accesser, increment) {
  using namespace NeuralNet;
  using namespace std;

  Shape shape({3, 1, 2});
  Accesser accesser(&shape);

  Address address(3, 0);

  EXPECT_EQ(address, std::vector<size_t>({0, 0, 0}));

  address = accesser.increment(std::move(address));
  EXPECT_EQ(address, std::vector<size_t>({0, 0, 1}));

  address = accesser.increment(std::move(address));
  EXPECT_EQ(address, std::vector<size_t>({1, 0, 0}));

  address = accesser.increment(std::move(address));
  EXPECT_EQ(address, std::vector<size_t>({1, 0, 1}));

  address = accesser.increment(std::move(address));
  EXPECT_EQ(address, std::vector<size_t>({2, 0, 0}));

  address = accesser.increment(std::move(address));
  EXPECT_EQ(address, std::vector<size_t>({2, 0, 1}));

  address = Address(3, 0);
  auto step = 2ul;
  address = accesser.increment(std::move(address), step);
  EXPECT_EQ(address, std::vector<size_t>({1, 0, 0}));

  address = accesser.increment(std::move(address), step);
  EXPECT_EQ(address, std::vector<size_t>({2, 0, 0}));
}

int main(int argc, char** argv) {
  // Disables elapsed time by default.
  ::testing::GTEST_FLAG(print_time) = false;

  // This allows the user to override the flag on the command line.
  ::testing::InitGoogleTest(&argc, argv);

  google::InstallFailureSignalHandler();

  return RUN_ALL_TESTS();
}
